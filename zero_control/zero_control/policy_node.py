"""Trained checkpoint -> /zero/eef_target. The teleop node's slot, driven by the policy.

    bash scripts/run_policy.sh [CHECKPOINT] [N_ACTION_STEPS]
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float64MultiArray

from zero_control.force import ToolFrameForce

from zero_control.action import DIM


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("zero_policy")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("dataset_root", "")
        self.declare_parameter("repo_id", "zero/base")
        # Empty means "take it from the dataset". The language instruction is part of the conditioning,
        # so a mismatch between training and inference is silent degradation that looks like a policy
        # failure. The default here was v1's string ("pick up the can and place it in the tray") while
        # cross_v1 was recorded with a longer one that names the handover. Reading it from the metadata
        # removes the whole class of bug.
        self.declare_parameter("task", "")
        self.declare_parameter("rate_hz", 10.0)          # must equal the dataset fps
        self.declare_parameter("n_action_steps", 25)     # replan horizon; <= chunk_size
        self.declare_parameter("cameras", ["front", "left_wrist", "right_wrist"])
        self.declare_parameter("autostart", False)
        self.declare_parameter("residual_warn_m", 0.02)
        # Dump every tick to an npz for offline analysis. The closed loop cannot be reproduced offline,
        # since the images depend on where the arm actually went, so when a rollout misbehaves this trace
        # is the only way to see the measured state and the commanded action side by side over time.
        self.declare_parameter("trace_path", "")
        # Mirage uses a "high-gain or blocking controller on the target robot" and requires the norm of
        # the error from the desired pose to be < 0.015 m at every timestep. Our servo is a one-step DLS
        # chase: a 70-86 mm move needs ~20 ticks (200 ms) to settle but gets 100 ms at 10 Hz, so it never
        # catches up, which is the 22-41 mm residual seen in rollout. Blocking decouples policy rate from
        # settling time. It makes the rollout slower than real time, which is fine in sim, and it is what
        # makes `action` equal to the achieved pose, so no forward dynamics model is needed.
        self.declare_parameter("block_until_settled", True)
        self.declare_parameter("settle_m", 0.015)      # Mirage's position tolerance
        # Mirage bounds the norm of the pose error including the quaternion, so orientation has to gate
        # the wait too. /zero/ik_status carries rot_err at indices 1 and 4.
        self.declare_parameter("settle_rad", 0.03)     # ~1.7 deg
        self.declare_parameter("settle_timeout_s", 1.0)
        # Blocking and a short n_action_steps work against each other: at 5 steps the target moves every
        # 0.5 s and the arm cannot settle inside the tolerance, giving 64 stalls in 106 s. The short
        # horizon did not help anyway (closest approach 61.5 mm against 57.0 mm at 25 steps), so prefer 25
        # and leave blocking on.

        ckpt = self.get_parameter("checkpoint").value
        if not ckpt or not Path(ckpt).exists():
            raise SystemExit(f"checkpoint '{ckpt}' not found. Pass -p checkpoint:=<dir>")
        self.task = self.get_parameter("task").value or None
        self.cams = list(self.get_parameter("cameras").value)
        self.warn_m = float(self.get_parameter("residual_warn_m").value)
        self.blocking = bool(self.get_parameter("block_until_settled").value)
        self.settle_m = float(self.get_parameter("settle_m").value)
        self.settle_rad = float(self.get_parameter("settle_rad").value)
        self.settle_timeout = float(self.get_parameter("settle_timeout_s").value)
        self.residual = 0.0
        self.residual_rot = 0.0
        self.waiting_since = None
        self.stalls = 0
        self._last_warn = 0.0
        self.trace_path = self.get_parameter("trace_path").value
        self.trace: list[np.ndarray] = []

        self.policy, self.pre, self.post, horizon = self._load(ckpt)

        self.state: np.ndarray | None = None
        # Dump the ACTUAL observation at the handover, so appearance questions can be settled by
        # looking rather than by reasoning about geometry. The trace records state and action only,
        # which was enough to rule out position (the vx300s reaches the same handover pose as a
        # successful reBot run, 82 mm vs 81 mm from the training mean) but cannot say anything about
        # what the cameras saw. Fires when the giving hand is closed and the two hands are close,
        # which is the moment the receiving hand has to decide.
        self.declare_parameter("camera_ns", "/zero")
        ToolFrameForce.declare(self)
        self.declare_parameter("frame_dump_dir", "")
        self.declare_parameter("frame_dump_max", 24)
        # Dump every Nth tick as well as at the handover. A rollout that stalls before the hands
        # meet otherwise produces no frames at all, and there is nothing to compare with training.
        self.declare_parameter("frame_dump_every", 0)
        self.dump_dir = str(self.get_parameter("frame_dump_dir").value)
        self.dump_max = int(self.get_parameter("frame_dump_max").value)
        self.dump_every = int(self.get_parameter("frame_dump_every").value)
        self.dumped = 0
        if self.dump_dir:
            import os
            os.makedirs(self.dump_dir, exist_ok=True)
        self.rgb: dict[str, np.ndarray] = {}
        self.running = bool(self.get_parameter("autostart").value)
        self.prev_x = 0
        self.infer_ms = 0.0
        self.replans = 0
        self.ticks = 0

        # Where the images come from. Point this at the shadow renderer's namespace to feed the
        # policy the SOURCE robot's rendering instead of the target's own cameras; the rest of the
        # node is unchanged, so switching is a launch argument and switching back is the revert.
        cam_ns = self.get_parameter("camera_ns").value.rstrip("/")
        if cam_ns != "/zero":
            self.get_logger().info(f"reading camera images from {cam_ns}/* (cross-painted)")
        for c in self.cams:
            self.create_subscription(Image, f"{cam_ns}/{c}/image_raw",
                                     lambda m, c=c: self._on_rgb(m, c), 10)
        # Force is optional: a robot with no F/T sensors configured yields an empty list, and a v1
        # checkpoint ignores the key anyway. A v2 checkpoint without it runs degraded, so log which.
        self.force = ToolFrameForce(self) if self.get_parameter("ft_sensors").value != [""] else None
        if self.force is None or not self.force.sensors:
            self.force = None
            self.get_logger().warn(
                "no F/T sensors configured; observation.force will be absent. A v2 checkpoint "
                "trained with force will run without one of its inputs.")
        else:
            self.get_logger().info(
                f"force observation from {len(self.force.sensors)} F/T sensors, "
                f"normalised by grip_force={self.force.grip_force:.1f} N")

        self.create_subscription(Float64MultiArray, "/zero/eef_state", self._on_state, 10)
        self.create_subscription(Float64MultiArray, "/zero/ik_status", self._on_status, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/zero/eef_target", 10)

        # Refuse to run against a competing publisher. teleop_node publishes the same topic at 50 Hz
        # and, with no stick input, holds the home pose, so a teleop left running from a recording session
        # overwrites every policy command 5x per tick. The policy asked for 162 mm of motion and the arm
        # moved 2.3 mm in 106 s, while `resid` read 1 mm because the arm was tracking teleop's home target
        # perfectly. That is invisible from this node's own logs, which is why it needs a guard rather
        # than a comment.
        others = [info.node_name
                  for info in self.get_publishers_info_by_topic("/zero/eef_target")
                  if info.node_name != self.get_name()]
        if others:
            raise SystemExit(
                f"\n/zero/eef_target already has publisher(s): {others}.\n"
                f"They will fight this node for control of the arm; teleop publishes at 50 Hz\n"
                f"against this node's {self.get_parameter('rate_hz').value:.0f} Hz and holds the "
                f"home pose when idle.\n  Stop it first:  pkill -f 'zero_control/teleop'\n")

        hz = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f"policy ready: {Path(ckpt).parents[2].name}/{Path(ckpt).parents[0].name} | "
            f"{len(self.cams)} cameras | {hz:.0f} Hz "
            f"| replan every {horizon} steps ({horizon/hz:.1f} s) | task={self.task!r} | "
            f"{'RUNNING' if self.running else 'press X to start'}")

    # ------------------------------------------------------------------ policy
    @staticmethod
    def _register_zerovla() -> None:
        """Teach LeRobot's factory about our policy type.

        `factory.get_policy_class` is a hardcoded if/elif chain, so a checkpoint whose config says
        `type: zerovla` cannot be loaded by stock LeRobot: it fails to parse its own checkpoint.
        Importing the module registers the config subclass; replacing the module attribute covers
        the class lookup. Same two lines as scripts/train_v2.py, for the same reason.
        """
        import lerobot.policies.factory as factory
        from zero_control.v2.policy import ZeroVLAPolicy

        if getattr(factory, "_zero_patched", False):
            return
        original = factory.get_policy_class

        def get_policy_class(name: str):
            return ZeroVLAPolicy if name == "zerovla" else original(name)

        factory.get_policy_class = get_policy_class
        factory._zero_patched = True

    def _load(self, ckpt: str):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self._register_zerovla()

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        cfg.device = "cuda"
        # A shorter horizon than training is an inference choice, and a good one: open-loop chunk
        # error was measured growing from 31 mm at k=0 to 47 mm at k=49, so replanning sooner is
        # strictly better. Note this CLAMPS to the trained chunk size, so asking for more than
        # chunk_size silently does nothing.
        cfg.n_action_steps = min(int(self.get_parameter("n_action_steps").value), cfg.chunk_size)

        meta = LeRobotDatasetMetadata(self.get_parameter("repo_id").value,
                                      root=self.get_parameter("dataset_root").value or None)
        if self.task is None:
            tasks = list(meta.tasks.index)
            if len(tasks) != 1:
                raise SystemExit(f"dataset has {len(tasks)} tasks {tasks}; pass -p task:=<one>")
            self.task = tasks[0]
            self.get_logger().info(f"task taken from the dataset: {self.task!r}")
        policy = make_policy(cfg=cfg, ds_meta=meta)
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=ckpt)
        policy.eval()
        return policy, pre, post, cfg.n_action_steps

    def _maybe_dump(self, a: np.ndarray) -> None:
        """Save the images and state the policy just acted on.

        Two triggers. The handover one fires once a hand is closed and the two are within 150 mm,
        which is the moment the receiving hand has to decide. That is useless for a rollout that
        stalls earlier, so `frame_dump_every` also dumps every Nth tick, giving frames from the
        approach and the grasp to compare against the training video at the same phase.
        """
        if not self.dump_dir or self.dumped >= self.dump_max or self.state is None:
            return
        giving_closed = a[9] < 0.5 or a[19] < 0.5
        gap = float(np.linalg.norm(np.asarray(a[0:3]) - np.asarray(a[10:13])))
        at_handover = giving_closed and gap < 0.15
        periodic = self.dump_every > 0 and self.ticks % self.dump_every == 0
        if not (at_handover or periodic):
            return
        tag = "handover" if at_handover else "frame"
        import os
        np.savez_compressed(
            os.path.join(self.dump_dir, f"{tag}_{self.ticks:05d}.npz"),
            state=np.asarray(self.state, dtype=np.float32),
            action=a.astype(np.float32),
            gap=np.float32(gap),
            **{c: self.rgb[c] for c in self.cams if c in self.rgb})
        self.dumped += 1
        if self.dumped == 1:
            self.get_logger().info(
                f"dumping handover observations to {self.dump_dir} (hands {gap*1000:.0f} mm apart)")

    def _tick(self) -> None:
        import torch
        if not self.running or self.state is None or len(self.rgb) < len(self.cams):
            return
        if self.blocking and self.ticks and (self.residual > self.settle_m
                                             or self.residual_rot > self.settle_rad):
            # Hold the current target until the arm arrives, rather than issuing a new one the servo has no
            # chance of reaching. Timeout so an unreachable pose cannot deadlock.
            if self.waiting_since is None:
                self.waiting_since = time.perf_counter()
            if time.perf_counter() - self.waiting_since < self.settle_timeout:
                return
            self.stalls += 1
            self.get_logger().warn(
                f"target not settled after {self.settle_timeout:.1f}s "
                f"({self.residual*1000:.0f} mm, {np.degrees(self.residual_rot):.1f} deg), "
                f"advancing anyway (stall #{self.stalls})")
        self.waiting_since = None
        batch = {"observation.state": torch.from_numpy(self.state).unsqueeze(0)}
        if self.force is not None:
            # The v2 policy trains on observation.force. Omitting it here does NOT raise: the key
            # is simply absent, the force token is skipped, and the model silently runs on an input
            # set it never saw. Publish it whenever the sensors exist.
            batch["observation.force"] = torch.from_numpy(self.force.vector()).unsqueeze(0)
        for c in self.cams:
            img = self.rgb[c].astype(np.float32) / 255.0                  # HWC RGB [0,1]
            batch[f"observation.images.{c}"] = torch.from_numpy(
                np.transpose(img, (2, 0, 1))).unsqueeze(0)                # -> 1,C,H,W
        batch["task"] = [self.task]

        t0 = time.perf_counter()
        with torch.no_grad():
            action = self.post(self.policy.select_action(self.pre(batch)))
        # Only 1 tick in n_action_steps runs the flow-matching integration; the rest pop a queued action
        # in ~0.1 ms. Averaging both together reported "9 ms" for what is really a ~225 ms inference, so
        # track the expensive calls only.
        dt = (time.perf_counter() - t0) * 1000
        if dt > 20.0:
            self.infer_ms = dt
            self.replans += 1

        a = action[0, :DIM].float().cpu().numpy().astype(float)
        self._maybe_dump(a)
        self.pub.publish(Float64MultiArray(data=a.tolist()))
        if self.trace_path:
            self.trace.append(np.concatenate([[self.ticks, self.replans, self.residual,
                                               self.residual_rot], self.state, a]))
        self.ticks += 1
        if self.ticks % 50 == 0:
            self.get_logger().info(
                f"step {self.ticks}  replans {self.replans} (last {self.infer_ms:.0f} ms)  "
                f"L {np.round(a[0:3], 3)}  "
                f"R {np.round(a[10:13], 3)}  grip {a[9]:.2f}/{a[19]:.2f}  "
                f"resid {self.residual*1000:.0f} mm/{np.degrees(self.residual_rot):.1f}deg  "
                f"stalls {self.stalls}")

    # ------------------------------------------------------------------ ROS in
    def _on_rgb(self, msg: Image, cam: str) -> None:
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.rgb[cam] = img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()

    def _on_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == DIM:
            self.state = np.asarray(msg.data, dtype=np.float32)

    def _on_status(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 6:
            return
        self.residual = max(msg.data[0], msg.data[3])       # worst arm gates the wait
        self.residual_rot = max(msg.data[1], msg.data[4])
        # Throttle on time, not on self.ticks: ticks stops incrementing while blocking waits, so a
        # tick-based gate fires on every 100 Hz status message and floods the log.
        now = time.perf_counter()
        if now - self._last_warn < 2.0:
            return
        self._last_warn = now
        for side, err in (("left", msg.data[0]), ("right", msg.data[3])):
            if err > self.warn_m:
                self.get_logger().warn(
                    f"{side} IK residual {err*1000:.0f} mm; the arm is stalling short of the "
                    f"commanded pose; this is NOT the policy failing")

    def _on_joy(self, msg: Joy) -> None:
        x = msg.buttons[2] if len(msg.buttons) > 2 else 0
        if x and not self.prev_x:
            self.running = not self.running
            if self.running:
                self.policy.reset()          # clear the action queue between attempts
                self.ticks = 0
            self.get_logger().info("RUNNING" if self.running else "STOPPED")
        self.prev_x = x


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.trace_path and node.trace:
            np.savez(node.trace_path, trace=np.array(node.trace))
            print(f"wrote {node.trace_path}  ({len(node.trace)} ticks)")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
