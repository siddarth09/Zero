"""Record teleop demos straight into a LeRobotDataset.

    ros2 run zero_control record --ros-args \
        --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml \
        -p root:=$HOME/zero_data/rebot_pick_place \
        -p task:="pick up the can and place it in the tray"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, Joy
from std_msgs.msg import Float64MultiArray

from zero_control.action import DIM, SIDES, pack
from zero_control.ik import ArmIK


class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("zero_record")
        self.declare_parameter("robot", "rebot")
        self.declare_parameter("root", "")
        self.declare_parameter("task", "pick up the can and place it in the tray")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("depth", True)
        self.declare_parameter("cameras", ["front", "left_wrist", "right_wrist"])
        self.declare_parameter("ft_sensors", [""])
        self.declare_parameter("ft_rot", [0.0])       # 9 per sensor, sensor frame -> tool frame
        self.declare_parameter("ft_side", [""])       # which hand each sensor belongs to
        self.declare_parameter("grip_force", 1.0)     # N, for normalising the magnitude
        self.declare_parameter("ft_bias", 0.0)        # N, no-load reading, see force.assemble
        self.declare_parameter("eef_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("play_sounds", True)  # spoken episode prompts, as lerobot does
        self.declare_parameter("keyboard", True)     # also accept lerobot's key bindings
        # X, not BACK: reachable with a thumb without letting go of a stick. Free because teleop already
        # uses A (home), Y (grip), LB/RB (dead-man) and START (reseed).
        self.declare_parameter("btn_toggle", 2)      # X
        self.declare_parameter("btn_discard", 1)     # B
        for side in SIDES:
            self.declare_parameter(f"{side}_arm_joints", [""])
            self.declare_parameter(f"{side}_eef_frame", "")

        self.key = self.get_parameter("robot").value
        root = self.get_parameter("root").value
        if not root:
            raise SystemExit("pass -p root:=<dataset dir>")
        self.root = Path(root)
        self.task = str(self.get_parameter("task").value)
        self.fps = float(self.get_parameter("fps").value)
        self.cams = list(self.get_parameter("cameras").value)
        self.fts = [s for s in self.get_parameter("ft_sensors").value if s]
        self.use_depth = bool(self.get_parameter("depth").value)

        rot = np.asarray(self.get_parameter("ft_rot").value, dtype=float)
        self.ft_side = [x for x in self.get_parameter("ft_side").value if x]
        self.grip_force = max(float(self.get_parameter("grip_force").value), 1e-6)
        self.ft_bias = float(self.get_parameter("ft_bias").value)
        if self.fts:
            if rot.size != 9 * len(self.fts) or len(self.ft_side) != len(self.fts):
                raise SystemExit(
                    f"ft_rot/ft_side do not match {len(self.fts)} sensors "
                    f"({rot.size} rotation values, {len(self.ft_side)} sides). Regenerate "
                    f"the control yaml with scripts/gen_bringup.py")
            self.ft_rot = rot.reshape(len(self.fts), 3, 3)
        else:
            self.ft_rot = np.zeros((0, 3, 3))

        urdf = str(Path(get_package_share_directory("zero_description"))
                   / "urdf" / f"zero_{self.key}.urdf")
        off = tuple(self.get_parameter("eef_offset").value)
        self.ik = {s: ArmIK(urdf, self.get_parameter(f"{s}_eef_frame").value,
                            list(self.get_parameter(f"{s}_arm_joints").value), off)
                   for s in SIDES}
        self.arm_joints = {s: list(self.get_parameter(f"{s}_arm_joints").value) for s in SIDES}

        self.q = np.zeros(self.ik["left"].model.nq)
        self.joint_names: list[str] = []
        self.joint_pos = np.zeros(0, np.float32)
        self.joint_vel = np.zeros(0, np.float32)
        self.rgb: dict[str, np.ndarray | None] = {c: None for c in self.cams}
        self.dep: dict[str, np.ndarray | None] = {c: None for c in self.cams}
        self.ft: dict[str, np.ndarray] = {s: np.zeros(6, np.float32) for s in self.fts}
        self.action = np.zeros(DIM, np.float32)
        self.have_action = False
        self.ik_res = np.zeros(6, np.float32)
        self.have_js = False
        self.prev_buttons: list[int] = []
        self.joy: Joy | None = None

        self.ds = None                      # created lazily, once shapes are known
        self.recording = False
        self.n_frames = 0
        self.n_episodes = 0
        self.play_sounds = bool(self.get_parameter("play_sounds").value)
        self._log_say = None
        # Set by the keyboard listener thread, consumed by the timer. Plain bools written from one thread
        # and read from another need no lock here: each is a single latched edge, and losing a race would
        # at worst delay a keypress by one 10 Hz tick.
        self._kb = {"start": False, "save": False, "discard": False, "stop": False}
        self._listener = None
        if bool(self.get_parameter("keyboard").value):
            self._start_keyboard()

        for c in self.cams:
            self.create_subscription(Image, f"/zero/{c}/image_raw",
                                     lambda m, k=c: self._on_rgb(m, k), 5)
            if self.use_depth:
                self.create_subscription(Image, f"/zero/{c}/depth",
                                         lambda m, k=c: self._on_depth(m, k), 5)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        for s in self.fts:
            self.create_subscription(WrenchStamped, f"/{s}_broadcaster/wrench",
                                     lambda m, k=s: self._on_ft(m, k), 10)
        self.create_subscription(Float64MultiArray, "/zero/eef_target", self._on_action, 10)
        self.create_subscription(Float64MultiArray, "/zero/ik_status", self._on_status, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.create_timer(1.0 / self.fps, self._tick)
        self.get_logger().info(
            f"recorder ready: {self.fps:.0f} fps, {len(self.cams)} cameras, "
            f"{len(self.fts)} F/T, depth={'on' if self.use_depth else 'off'}\n"
            f"  dataset root: {self.root}\n"
            f"  task: {self.task!r}\n"
            "  gamepad: X = start/stop episode, B = discard\n"
            "  keyboard: SPACE = start, RIGHT = save, LEFT = discard, ESC = stop")
        self._say("Recorder ready")

    # ---------------------------------------------------------------- callbacks
    def _on_rgb(self, msg: Image, cam: str) -> None:
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.rgb[cam] = img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()

    def _on_depth(self, msg: Image, cam: str) -> None:
        if msg.encoding == "32FC1":
            d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        else:                                     # 16UC1 millimetres -> metres
            d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width) / 1000.0
        self.dep[cam] = d.astype(np.float32).copy()

    def _on_js(self, msg: JointState) -> None:
        model = self.ik["left"].model
        for name, pos in zip(msg.name, msg.position):
            jid = model.getJointId(name)
            if jid < model.njoints:
                self.q[model.joints[jid].idx_q] = pos
        if not self.joint_names:
            self.joint_names = list(msg.name)
        self.joint_pos = np.asarray(msg.position, np.float32)
        self.joint_vel = (np.asarray(msg.velocity, np.float32) if len(msg.velocity)
                          else np.zeros_like(self.joint_pos))
        self.have_js = True

    def _on_ft(self, msg: WrenchStamped, sensor: str) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        self.ft[sensor] = np.array([f.x, f.y, f.z, t.x, t.y, t.z], np.float32)

    def _on_action(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == DIM:
            self.action = np.asarray(msg.data, np.float32)
            self.have_action = True

    def _on_status(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == 6:
            self.ik_res = np.asarray(msg.data, np.float32)

    def _on_joy(self, msg: Joy) -> None:
        self.joy = msg

    def _pressed(self, which: str) -> bool:
        if self.joy is None:
            return False
        idx = int(self.get_parameter(f"btn_{which}").value)
        if not 0 <= idx < len(self.joy.buttons):
            return False
        was = self.prev_buttons[idx] if idx < len(self.prev_buttons) else 0
        return bool(self.joy.buttons[idx]) and not was

    # ---------------------------------------------------------------- keyboard
    def _start_keyboard(self) -> None:
        """lerobot's key bindings, on a background listener.

        pynput needs a graphical session, so this degrades to gamepad-only rather than failing --
        the same fallback lerobot makes. Note it grabs keys GLOBALLY, not just when this terminal
        has focus, which is what makes it usable while you are watching the sim window.
        """
        try:
            from pynput import keyboard
        except Exception as exc:
            self.get_logger().warn(
                f"keyboard control unavailable ({exc}); gamepad only")
            return

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self._kb["start"] = True
                elif key == keyboard.Key.right:
                    self._kb["save"] = True
                elif key == keyboard.Key.left:
                    self._kb["discard"] = True
                elif key == keyboard.Key.esc:
                    self._kb["stop"] = True
            except Exception as exc:                       # never let the listener thread die
                self.get_logger().warn(f"key handling failed: {exc}")

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            self.get_logger().info(
                "keyboard: SPACE start, RIGHT save, LEFT discard, ESC stop")
        except Exception as exc:
            self.get_logger().warn(f"could not start keyboard listener ({exc}); gamepad only")

    # ---------------------------------------------------------------- spoken prompts
    def _say(self, text: str) -> None:
        """Speak an episode prompt, the way lerobot_record does.

        Teleoperating means both hands on the pad and eyes on the sim, so a terminal line is the
        one place the operator is guaranteed not to be looking. lerobot's own `log_say` is reused
        rather than reimplemented, so phrasing and behaviour stay consistent with their tooling.

        Non-blocking, deliberately. This runs inside the 10 Hz recording timer, and a blocking
        `spd-say --wait` would stall the callback for the length of the utterance and drop frames
        from the episode it is announcing.
        """
        self.get_logger().info(text)
        if not self.play_sounds:
            return
        if self._log_say is None:
            try:
                from lerobot.utils.utils import log_say
                self._log_say = log_say
            except Exception:                     # lerobot is only needed for the dataset itself
                import subprocess

                def _fallback(t, play_sounds=True, blocking=False):
                    subprocess.Popen(["spd-say", t],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._log_say = _fallback
                self.get_logger().warn("lerobot log_say unavailable; using spd-say directly")
        try:
            self._log_say(text, play_sounds=True, blocking=False)
        except Exception as exc:
            self.get_logger().warn(f"speech failed ({exc}); continuing silently")
            self.play_sounds = False

    # ---------------------------------------------------------------- dataset
    def _features(self) -> dict:
        n_joint = len(self.joint_pos)
        feats: dict = {
            "observation.state": {"dtype": "float32", "shape": (DIM,), "names": None},
            # Policy-visible force: the F in VLFA. 7 per hand, wrench(6) plus normalised magnitude.
            "observation.force": {
                "dtype": "float32", "shape": (14,),
                "names": [f"{s}.{a}" for s in SIDES
                          for a in ("net_fx", "net_fy", "net_fz",
                                    "net_tx", "net_ty", "net_tz", "squeeze")]},
            "action": {"dtype": "float32", "shape": (DIM,), "names": None},
            "observation.joint_pos": {"dtype": "float32", "shape": (n_joint,),
                                      "names": self.joint_names},
            "observation.joint_vel": {"dtype": "float32", "shape": (n_joint,),
                                      "names": self.joint_names},
            "observation.ft": {"dtype": "float32", "shape": (6 * len(self.fts),),
                               "names": [f"{s}.{a}" for s in self.fts
                                         for a in ("fx", "fy", "fz", "tx", "ty", "tz")]},
            "observation.ik_residual": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.grip_cmd": {"dtype": "float32", "shape": (2,), "names": list(SIDES)},
        }
        for c in self.cams:
            h, w = self.rgb[c].shape[:2]
            feats[f"observation.images.{c}"] = {"dtype": "video", "shape": (h, w, 3),
                                               "names": ["height", "width", "channel"]}
            if self.use_depth:
                feats[f"observation.depth.{c}"] = {"dtype": "float32", "shape": (h, w),
                                                  "names": ["height", "width"]}
        return feats

    def _ensure_dataset(self) -> bool:
        """Create the dataset, or resume an existing one at the same root.

        `LeRobotDataset.create` raises FileExistsError on an existing root, so without this,
        restarting the recorder after a crash, a break or a deliberate stop could only ever start
        a fresh directory, and a 30-episode session had to be one unbroken process.

        Validate before loading, because `meta/info.json` is not enough. LeRobotDataset only
        writes `meta/episodes/*.parquet` when the writers are closed, so a session that was killed
        leaves `info.json` claiming N episodes with no episode index at all. Handing that to
        LeRobotDataset does not fail locally: it treats the missing metadata as "not downloaded
        yet" and goes to the Hugging Face Hub, which 404s for a dataset that exists only on disk.
        The traceback then points at huggingface_hub and hides the real problem, a half-written
        directory. Checking for the episode index up front turns that into one sentence. (An
        earlier note here blamed the 404 on a missing HF_HUB_OFFLINE; that was wrong. The hub is
        only consulted when the local metadata is incomplete.)

        The feature sets are compared before appending. The schema here has changed repeatedly
        (force promoted into the observation, depth added, joint velocities added), and appending
        frames whose keys disagree with the existing episodes would produce one dataset that is
        quietly two incompatible halves.
        """
        if self.ds is not None:
            return True
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # belt and braces; not the actual fix
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        want = self._features()
        threads = 2 * len(self.cams)
        info = self.root / "meta" / "info.json"
        episodes_idx = sorted((self.root / "meta" / "episodes").glob("**/*.parquet"))
        # Distinguish "never used" from "lost its index" before the check below. `create()` writes
        # meta/info.json immediately, so starting the recorder and stopping it before the first episode
        # leaves a skeleton with no episode index. That is indistinguishable from a dataset whose index was
        # lost, but it has nothing in it to protect, and refusing to record is pure friction that fires
        # every time the recorder is restarted between can positions.
        if (info.exists() and not episodes_idx
                and not sorted((self.root / "data").glob("**/*.parquet"))
                and not sorted((self.root / "videos").glob("**/*.mp4"))):
            import shutil
            shutil.rmtree(self.root)
            self.get_logger().info(
                f"{self.root} held an empty dataset skeleton (no frames, no videos) left by a "
                f"recorder stopped before its first episode. Recreating it.")
        if info.exists() and not episodes_idx:
            # Log, do not raise. This runs inside a timer callback, so a SystemExit propagates out of
            # rclpy.spin and lands in main's `except SystemExit`, which exists for the ESC path and swallows
            # the message. The recorder then died with nothing after "Recorder ready", which is worse than the
            # hub traceback it replaced.
            self.get_logger().error(
                f"\n{self.root} was created but never finalised: meta/info.json is there but "
                f"meta/episodes/ has no parquet, so the episode index is missing and the dataset "
                f"cannot be opened.\nThis is what a killed recorder leaves behind.\n"
                f"  Its frame data may still be readable, check with:\n"
                f"    python3 -c \"import glob,pyarrow.parquet as pq; "
                f"[print(f, pq.read_table(f).num_rows) for f in "
                f"glob.glob('{self.root}/data/**/*.parquet', recursive=True)]\"\n"
                f"  To carry on recording, move it aside and use a fresh root:\n"
                f"    mv {self.root} {self.root}.broken\n")
            self._say("Dataset is broken. Not recording")
            return False
        if info.exists():
            self.ds = LeRobotDataset(repo_id=f"zero/{self.key}", root=self.root)
            have = set(self.ds.meta.features)
            missing = set(want) - have
            extra = {k for k in have - set(want)
                     if k not in ("timestamp", "frame_index", "episode_index", "index",
                                  "task_index")}
            if missing or extra:
                self.get_logger().error(
                    f"cannot resume {self.root}: its schema does not match what this recorder "
                    f"would write.\n  missing from the dataset: {sorted(missing)}\n"
                    f"  present but not recorded now: {sorted(extra)}\n"
                    f"Record into a new root instead of mixing two schemas.")
                self.ds = None
                self._say("Schema mismatch. Not recording")
                return False
            if hasattr(self.ds, "start_image_writer") and self.cams:
                self.ds.start_image_writer(num_processes=0, num_threads=threads)
            self.n_episodes = int(self.ds.meta.total_episodes)
            self.get_logger().info(
                f"RESUMING dataset at {self.root}: {self.n_episodes} episodes already recorded")
            self._say(f"Resuming. {self.n_episodes} episodes on disk")
        else:
            self.ds = LeRobotDataset.create(
                repo_id=f"zero/{self.key}", fps=int(self.fps), features=want,
                root=self.root, robot_type=self.key,
                # Images go through worker threads; the control loop must not block on PNG/video encoding or the
                # dataset drops to the writer's rate.
                image_writer_processes=0, image_writer_threads=threads)
            self.get_logger().info(f"created dataset at {self.root}")
        return True

    def _tool_frame_force(self) -> np.ndarray:
        """Delegates to zero_control.force.assemble, which the policy node also uses.

        Kept as a one-line wrapper rather than inlined at the call site so the docstring above
        stays attached to the concept: the recorder and the policy must compute this identically,
        or the model is fed a quantity it never trained on with nothing raising.
        """
        from zero_control.force import assemble
        return assemble(self.fts, self.ft_side, self.ft_rot, self.grip_force, self.ft,
                        self.ft_bias)

    def _measured_poses(self) -> dict:
        """Measured end-effector pose per hand, by FK from the joints, not the commanded target."""
        return {side: (self.ik[side].fk(self.q).translation,
                       self.ik[side].fk(self.q).rotation) for side in SIDES}

    def _measured_state(self, poses: dict) -> np.ndarray:
        grips = {s: float(self.action[9 if s == "left" else 19]) for s in SIDES}
        return pack(poses, grips).astype(np.float32)

    def finalize(self) -> None:
        """Close the parquet writers. Without this the dataset is unreadable.

        LeRobotDataset buffers frames into a parquet writer that only stamps the file footer when
        it is closed, and `finalize()` is what closes it. Skip it and you get a plausible-looking
        directory (correct `meta/info.json`, correct episode and frame counts, a 33 MB data file)
        that pyarrow refuses to open with "Parquet magic bytes not found in footer". The frames
        are gone. A 2-episode, 97-frame take was lost that way.

        Note lerobot_record.py does not call this either; it gets away with it because
        `push_to_hub` closes the writers on the way out. Recording locally, nothing does, so this
        has to be explicit.

        Images are flushed first: the writer threads may still be encoding when the parquet
        closes.
        """
        if self.ds is None:
            return
        try:
            self.ds.stop_image_writer()
        except Exception as exc:
            self.get_logger().warn(f"stop_image_writer: {exc}")
        try:
            self.ds.finalize()
            self.get_logger().info(
                f"dataset finalised: {self.n_episodes} episodes at {self.root}. Safe to exit.")
            self._say("Dataset saved")
        except Exception as exc:
            self.get_logger().error(f"FINALIZE FAILED ({exc}). The parquet has no footer and "
                                    f"the episodes are not readable")

    # ---------------------------------------------------------------- loop
    def _tick(self) -> None:
        if self.joy is not None:
            if self._pressed("toggle"):
                if self.recording:
                    self._stop(save=True)
                else:
                    self._start()
            elif self._pressed("discard") and self.recording:
                self._stop(save=False)
            self.prev_buttons = list(self.joy.buttons)

        if self._kb["stop"]:
            self._kb["stop"] = False
            if self.recording:
                self._stop(save=True)
            self._say("Stop recording")
            self.finalize()
            raise SystemExit(0)
        if self._kb["discard"]:
            self._kb["discard"] = False
            if self.recording:
                self._stop(save=False)
        if self._kb["save"]:
            self._kb["save"] = False
            if self.recording:
                self._stop(save=True)
        if self._kb["start"]:
            self._kb["start"] = False
            if not self.recording:
                self._start()

        if not self.recording:
            return
        if not self.have_js or any(self.rgb[c] is None for c in self.cams):
            return
        if self.use_depth and any(self.dep[c] is None for c in self.cams):
            return

        measured = self._measured_poses()
        frame = {
            "observation.state": self._measured_state(measured),
            "observation.force": self._tool_frame_force(),
            "action": self.action.copy(),
            "observation.joint_pos": self.joint_pos.copy(),
            "observation.joint_vel": self.joint_vel.copy(),
            "observation.ft": np.concatenate([self.ft[s] for s in self.fts])
                              if self.fts else np.zeros(0, np.float32),
            "observation.ik_residual": self.ik_res.copy(),
            "observation.grip_cmd": np.array(
                [self.action[9], self.action[19]], np.float32),
            "task": self.task,
        }
        for c in self.cams:
            frame[f"observation.images.{c}"] = self.rgb[c]
            if self.use_depth:
                frame[f"observation.depth.{c}"] = self.dep[c]
        self.ds.add_frame(frame)
        self.n_frames += 1
        if self.n_frames % int(self.fps) == 0:
            self.get_logger().info(f"  recording... {self.n_frames} frames "
                                   f"({self.n_frames / self.fps:.1f} s)")

    def _start(self) -> None:
        if not self.have_js or any(self.rgb[c] is None for c in self.cams):
            self.get_logger().warn("not all streams up yet, not starting")
            return
        # Refuse to record without an action stream. Nothing else catches this: with the teleop node down,
        # /zero/eef_target is never published, `action` stays at its zero initialiser, and the recorder
        # writes a perfectly well-formed episode in which every action is the zero vector. It looks like a
        # real dataset, trains to nothing, and the only clue is a column of zeros nobody thinks to plot.
        if not self.have_action:
            self.get_logger().error(
                "no /zero/eef_target seen. Is the teleop node running? "
                "Refusing to record: every action would be zero.")
            return
        if not self._ensure_dataset():
            return
        self.recording = True
        self.n_frames = 0
        self._say(f"Recording episode {self.n_episodes}")

    def _stop(self, save: bool) -> None:
        self.recording = False
        if save and self.n_frames:
            self.ds.save_episode()
            self.n_episodes += 1
            self.get_logger().info(
                f"=== episode SAVED: {self.n_frames} frames, "
                f"{self.n_frames / self.fps:.1f} s  (total {self.n_episodes}) ===")
            # Two prompts, because they are two different instructions: the first confirms the take was kept,
            # the second is the operator's cue to put the can back.
            self._say(f"Episode saved. {self.n_episodes} recorded")
            self._say("Reset the environment")
        else:
            if self.ds is not None:
                self.ds.clear_episode_buffer()
            self._say("Episode discarded. Re-record")
        self.n_frames = 0


def main() -> None:
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.recording:
            node._stop(save=True)
        node._say("Stop recording")
        node.finalize()
    except SystemExit:
        pass
    finally:
        # finalize() is idempotent enough to call again: the ESC path and the Ctrl-C path both run it, and
        # a second close is a no-op. Losing the footer is far worse than closing twice.
        node.finalize()
        if node._listener is not None:
            node._listener.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
