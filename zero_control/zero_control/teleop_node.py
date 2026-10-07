"""Gamepad -> /zero/eef_target. Integrates stick velocity into an absolute SE(3) target.

    ros2 run joy joy_node --ros-args -p deadzone:=0.05 -p autorepeat_rate:=20.0
    ros2 run zero_control teleop --ros-args \
        --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml \
        --params-file install/zero_bringup/share/zero_bringup/config/teleop.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float64MultiArray

from zero_control.action import SIDES, pack
from zero_control.ik import ArmIK


def _yaw(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("zero_teleop")
        self.declare_parameter("robot", "rebot")
        self.declare_parameter("eef_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("lin_speed", 0.12)       # m/s at full stick
        self.declare_parameter("yaw_speed", 0.9)        # rad/s at full stick
        self.declare_parameter("expo", 2.0)             # stick response curve
        self.declare_parameter("grip_speed", 0.7)       # fraction of full travel per second
        # 0.03 -> 0.015 on 2026-08-26. The leash bounds how far the commanded target may lead the measured
        # pose, so it directly bounds the gap between `action` and `observation.state` in the recorded
        # dataset. At 0.03 the v1 recording had 28% of frames more than 15 mm from their commanded pose
        # (p90 29.5 mm, max 31.5 mm, exactly the leash). Mirage requires the achieved pose to be within
        # 0.015 m of the desired pose at every timestep, and that is also what makes `action` usable as
        # "the next pose to achieve" without fitting a forward dynamics model f(p, a) -> p'. 0.015 makes
        # the dataset satisfy that by construction.
        self.declare_parameter("leash", 0.015)          # m the target may lead the arm
        self.declare_parameter("leash_rot", 0.05)       # rad, same idea as `leash` above
        self.declare_parameter("ws_min", [-1.0, -1.0, 0.0])
        self.declare_parameter("ws_max", [1.0, 1.0, 2.0])
        # Axis and button map. Defaults are the standard XInput layout, which is what joy_node reports
        # for this pad; `ros2 run zero_control joy_probe` prints live values to check.
        self.declare_parameter("axis_x", 1)             # left stick Y -> +x (forward)
        self.declare_parameter("axis_y", 0)             # left stick X -> +y
        self.declare_parameter("axis_z", 4)             # right stick Y -> +z
        self.declare_parameter("axis_yaw", 3)           # right stick X -> yaw
        self.declare_parameter("sign_x", 1.0)
        self.declare_parameter("sign_y", 1.0)
        self.declare_parameter("sign_z", 1.0)
        self.declare_parameter("sign_yaw", 1.0)
        # Analog grip, disabled by default (-1). See the module docstring.
        self.declare_parameter("axis_grip_left", -1)
        self.declare_parameter("axis_grip_right", -1)
        self.declare_parameter("btn_left", 4)           # LB dead-man, left arm
        self.declare_parameter("btn_right", 5)          # RB dead-man, right arm
        self.declare_parameter("btn_grip", 3)           # Y  -> toggle the selected gripper
        self.declare_parameter("btn_home", 0)           # A  -> back to home
        self.declare_parameter("btn_reseed", 7)         # Start -> resync target to measurement
        for side in SIDES:
            self.declare_parameter(f"{side}_arm_joints", [""])
            self.declare_parameter(f"{side}_eef_frame", "")
            self.declare_parameter(f"{side}_home", [0.0])

        key = self.get_parameter("robot").value
        urdf = str(Path(get_package_share_directory("zero_description"))
                   / "urdf" / f"zero_{key}.urdf")
        off = tuple(self.get_parameter("eef_offset").value)
        self.ws_min = np.asarray(self.get_parameter("ws_min").value, dtype=float)
        self.ws_max = np.asarray(self.get_parameter("ws_max").value, dtype=float)

        self.ik: dict[str, ArmIK] = {}
        self.home_pose: dict[str, pin.SE3] = {}
        for side in SIDES:
            jn = list(self.get_parameter(f"{side}_arm_joints").value)
            frame = self.get_parameter(f"{side}_eef_frame").value
            if not jn or not jn[0] or not frame:
                raise SystemExit(f"missing params for {side}: pass {key}_control.yaml")
            self.ik[side] = ArmIK(urdf, frame, jn, off)
            qh = np.zeros(self.ik[side].model.nq)
            qh[self.ik[side].qidx] = np.asarray(
                self.get_parameter(f"{side}_home").value, dtype=float)
            self.home_pose[side] = self.ik[side].fk(qh)

        self.q = np.zeros(self.ik["left"].model.nq)
        self.have_js = False
        self.joy: Joy | None = None
        self.pos: dict[str, np.ndarray] = {}
        self.rot: dict[str, np.ndarray] = {}
        self.grip = {s: 1.0 for s in SIDES}
        self.latched = {s: False for s in SIDES}   # False = open
        self.active_side = "left"                  # sticky, so Y works with no dead-man held
        self.seeded = False
        self.homing = False
        self.prev_buttons: list[int] = []

        self.pub = self.create_publisher(Float64MultiArray, "/zero/eef_target", 10)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"teleop robot={key} {hz:.0f} Hz  lin={self.get_parameter('lin_speed').value} m/s  "
            f"yaw={self.get_parameter('yaw_speed').value} rad/s\n"
            "  LB = drive LEFT arm, RB = drive RIGHT arm (nothing held = frozen)\n"
            "  left stick = X/Y   right stick = Z (up/down) + yaw (left/right)\n"
            "  Y  = toggle the selected arm's gripper (latches closed until pressed again)\n"
            "  A = return to home    START = resync target to the measured pose")

    # ------------------------------------------------------------------ callbacks
    def _on_js(self, msg: JointState) -> None:
        model = self.ik["left"].model
        for name, position in zip(msg.name, msg.position):
            jid = model.getJointId(name)
            if jid < model.njoints:
                self.q[model.joints[jid].idx_q] = position
        self.have_js = True
        if not self.seeded:
            self._seed()

    def _on_joy(self, msg: Joy) -> None:
        self.joy = msg

    def _seed(self) -> None:
        """Start from where the arm actually IS, so nothing jumps on the first stick input."""
        for side in SIDES:
            T = self.ik[side].fk(self.q)
            self.pos[side] = T.translation.copy()
            self.rot[side] = T.rotation.copy()
        self.seeded = True
        self.get_logger().info(
            "seeded from measured pose: "
            + "  ".join(f"{s}={np.round(self.pos[s], 3)}" for s in SIDES))

    # ------------------------------------------------------------------ helpers
    def _axis(self, name: str) -> float:
        """Signed, expo-shaped stick value. Expo keeps small corrections fine near centre."""
        if self.joy is None:
            return 0.0
        idx = int(self.get_parameter(f"axis_{name}").value)
        if not 0 <= idx < len(self.joy.axes):
            return 0.0
        v = float(self.joy.axes[idx]) * float(self.get_parameter(f"sign_{name}").value)
        e = float(self.get_parameter("expo").value)
        return float(np.sign(v) * abs(v) ** e)

    def _held(self, name: str) -> bool:
        if self.joy is None:
            return False
        idx = int(self.get_parameter(f"btn_{name}").value)
        return 0 <= idx < len(self.joy.buttons) and bool(self.joy.buttons[idx])

    def _pressed(self, name: str) -> bool:
        """Rising edge, so a held button fires once."""
        if self.joy is None:
            return False
        idx = int(self.get_parameter(f"btn_{name}").value)
        if not 0 <= idx < len(self.joy.buttons):
            return False
        was = self.prev_buttons[idx] if idx < len(self.prev_buttons) else 0
        return bool(self.joy.buttons[idx]) and not was

    def _trigger(self, side: str) -> float:
        """Analog trigger as 0 (released) .. 1 (fully pressed). Rests at +1 on XInput."""
        if self.joy is None:
            return 0.0
        idx = int(self.get_parameter(f"axis_grip_{side}").value)
        if not 0 <= idx < len(self.joy.axes):
            return 0.0
        return float(np.clip((1.0 - self.joy.axes[idx]) * 0.5, 0.0, 1.0))

    # ------------------------------------------------------------------ control loop
    def _tick(self) -> None:
        if not (self.have_js and self.seeded):
            return

        if self._pressed("home"):
            # A commanded motion, not a teleport. Assigning the home pose straight into the target does
            # not survive the leash below: the leash reels the target back to within `leash` of the arm
            # and overwrites it, so the home pose is gone after one tick and what remains is a 3 cm nudge
            # in whatever direction the first clamp happened to point. The arm then drifts instead of
            # going home, 200 mm off, worse than not pressing it. So home is a latched mode that walks the
            # target home at the same rate the sticks would, which the leash follows happily.
            self.homing = True
            self.get_logger().info("-> homing")
        if self._pressed("reseed"):
            self._seed()
            self.homing = False

        if self.homing:
            step = float(self.get_parameter("lin_speed").value) * self.dt
            done = True
            for side in SIDES:
                d = self.home_pose[side].translation - self.pos[side]
                n = float(np.linalg.norm(d))
                if n > 1e-4:
                    self.pos[side] = self.pos[side] + d * min(1.0, step / n)
                    done = done and n < 2e-3
                rerr = pin.log3(self.rot[side].T @ self.home_pose[side].rotation)
                a = float(np.linalg.norm(rerr))
                if a > 1e-4:
                    rstep = float(self.get_parameter("yaw_speed").value) * self.dt
                    self.rot[side] = self.rot[side] @ pin.exp3(rerr * min(1.0, rstep / a))
                    done = done and a < 5e-3
            if done:
                self.homing = False
                self.get_logger().info("at home")

        # Dead-man picks the arm. Nothing held = nothing integrates.
        active = None
        if self._held("left"):
            active = "left"
        elif self._held("right"):
            active = "right"
        if active is not None:
            self.active_side = active

        if self._pressed("grip"):
            side = self.active_side
            self.latched[side] = not self.latched[side]
            self.get_logger().info(
                f"{side} gripper {'closing' if self.latched[side] else 'opening'}")

        if active is not None:
            self.homing = False          # operator input always overrides the return-to-home
            lin = float(self.get_parameter("lin_speed").value) * self.dt
            yaw = float(self.get_parameter("yaw_speed").value) * self.dt
            self.pos[active] = np.clip(
                self.pos[active] + lin * np.array(
                    [self._axis("x"), self._axis("y"), self._axis("z")]),
                self.ws_min, self.ws_max)
            dyaw = yaw * self._axis("yaw")
            if dyaw:
                # World-frame yaw: pre-multiply, so left/right on the stick always means the same direction
                # on screen regardless of how the wrist is currently oriented.
                self.rot[active] = _yaw(dyaw) @ self.rot[active]

        # Leash. The target may never lead the measured pose by more than `leash`. The workspace box
        # alone is not enough: it is the table, but an arm's reach is smaller than the table, so
        # driving near a corner lets the target sail on into space the arm cannot follow. Without
        # this, a 2 s push commanded 240 mm, the arm managed 182 mm, and the target sat 58 mm ahead,
        # so releasing the stick did not stop the arm and it was still coasting while the other arm
        # was being driven, which reads like cross-talk. Reeling the target in each tick makes the
        # stick 1:1 (release and it stops; push into a reach limit and it stops instead of banking
        # travel you must unwind) and keeps every recorded action one the arm actually achieved, since
        # an unreachable target would be a mislabelled demo.
        for side in SIDES:
            cur = self.ik[side].fk(self.q)
            d = self.pos[side] - cur.translation
            n = float(np.linalg.norm(d))
            lead = float(self.get_parameter("leash").value)
            if n > lead:
                self.pos[side] = cur.translation + d * (lead / n)
            rerr = pin.log3(cur.rotation.T @ self.rot[side])
            a = float(np.linalg.norm(rerr))
            lead_r = float(self.get_parameter("leash_rot").value)
            if a > lead_r:
                self.rot[side] = cur.rotation @ pin.exp3(rerr * (lead_r / a))

        # Rate-limited grip. The latch is a boolean, and feeding it straight through steps the command
        # from 1.0 to 0.0 in one tick, so a position actuator closes as fast as its force limit allows.
        # Ramping instead took the reBot's lift from 64 mm to 84 mm, a materially more secure hold. The
        # reason is not that a fast close swats the object away: the object moves under 0.5 mm during
        # closing either way, and ramping actually moves it marginally more. Panda is unchanged at
        # ~92 mm, so this buys nothing there. 0.5 s measured as good as 1.4 s, so grip_speed can be
        # raised if the ramp feels sluggish. Opening ramps too: symmetric is one number to tune, and a
        # slow release lets a placed object settle rather than being flicked.
        rate = float(self.get_parameter("grip_speed").value) * self.dt
        for side in SIDES:
            idx = int(self.get_parameter(f"axis_grip_{side}").value)
            if idx >= 0:
                # Analog trigger: the operator's finger is the rate limit, so pass it through.
                self.grip[side] = 1.0 - self._trigger(side)
            else:
                goal = 0.0 if self.latched[side] else 1.0
                self.grip[side] += float(np.clip(goal - self.grip[side], -rate, rate))

        self.pub.publish(Float64MultiArray(data=[
            float(v) for v in pack({s: (self.pos[s], self.rot[s]) for s in SIDES}, self.grip)]))
        if self.joy is not None:
            self.prev_buttons = list(self.joy.buttons)


def main() -> None:
    rclpy.init()
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
