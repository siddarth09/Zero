"""Render the SOURCE robot doing what the TARGET robot is doing."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

import mujoco

from zero_control.action import unpack

SIDES = ("left", "right")


class ShadowRender(Node):
    def __init__(self) -> None:
        super().__init__("zero_shadow_render")
        # Everything model-specific arrives as a parameter, the same convention eef_control_node
        # follows, so this runs identically from source or an install tree and zero_layout stays the
        # single source of truth rather than being imported at runtime.
        self.declare_parameter("mjcf", "")
        self.declare_parameter("arm_joints", ["joint1", "joint2", "joint3",
                                              "joint4", "joint5", "joint6"])
        self.declare_parameter("namespace", "/shadow")
        self.declare_parameter("cameras", ["front", "left_wrist", "right_wrist"])
        self.declare_parameter("resolution", [224, 224])
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("table_top_z", 0.75)
        self.declare_parameter("can_x", 0.34)
        self.declare_parameter("can_y", 0.45)
        # The launch writes the can's real pose into this file to hand it to the simulator. Reading
        # it here means the shadow cannot disagree with the sim about where the object is. Passing
        # can_x/can_y to two processes by hand looks harmless and is not: a 57 mm mismatch renders
        # the can somewhere it is not, the policy flies to the drawn can, the visual signal never
        # strengthens, and the rollout stalls in a way that looks like a policy failure.
        self.declare_parameter("start_override", "/tmp/zero_vx300s_start.xml")
        # The override's qpos is laid out for the TARGET model, not this one, so finding the can's
        # slot needs the target's own joint layout. Guessing by value does not work: the tray has a
        # free joint at a similar height and gets picked instead.
        self.declare_parameter("target_mjcf", "")
        # Where a released can comes to rest. The tray does not move in this task, so its surface is
        # a task constant rather than something to measure: geom_rbound is a bounding SPHERE radius,
        # which overestimated the tray top by 230 mm, and its geoms are meshes with no usable
        # half-extents. Defaults match PLACE_POS in zero_layout.
        self.declare_parameter("tray_x", 0.34)
        self.declare_parameter("tray_y", -0.45)
        self.declare_parameter("tray_top_z", 0.765)
        self.declare_parameter("tray_radius", 0.14)
        self.declare_parameter("grip_closed", 0.5)
        # Above this the shadow arm is too far from the pose it was asked for to be showing the
        # truth. Publishing anyway would feed the policy a confident lie, so hold the last good
        # frame and say so instead.
        self.declare_parameter("max_residual_mm", 25.0)

        path = self.get_parameter("mjcf").value
        if not path:
            raise RuntimeError("shadow_render needs -p mjcf:=<path to the SOURCE robot's MJCF>")
        arm_joints = list(self.get_parameter("arm_joints").value)
        table_z = float(self.get_parameter("table_top_z").value)
        self.ns = self.get_parameter("namespace").value
        self.cams = list(self.get_parameter("cameras").value)
        self.grip_closed = float(self.get_parameter("grip_closed").value)
        self.max_res = float(self.get_parameter("max_residual_mm").value) / 1000.0

        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        w, h = (int(v) for v in self.get_parameter("resolution").value)
        self.renderer = mujoco.Renderer(self.model, h, w)

        self.arm = {}
        for side in SIDES:
            jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}")
                   for j in arm_joints]
            if any(j < 0 for j in jid):
                raise RuntimeError(f"{side} arm joints {arm_joints} are not all in {path}")
            self.arm[side] = {
                "jid": jid,
                "qadr": [self.model.jnt_qposadr[j] for j in jid],
                "dof": [self.model.jnt_dofadr[j] for j in jid],
                "site": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef"),
            }

        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_can")
        self.can_body = int(self.model.geom_bodyid[gid])
        self.table_z = table_z
        self.can_qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.can_body]]
        self.can_half = float(self.model.geom_size[gid][1])
        self.can_xy = (float(self.get_parameter("can_x").value),
                       float(self.get_parameter("can_y").value))
        override = str(self.get_parameter("start_override").value)
        found = self._can_from_override(override, str(self.get_parameter("target_mjcf").value))
        if found is not None:
            if np.linalg.norm(np.subtract(found, self.can_xy)) > 0.005:
                self.get_logger().warn(
                    f"can_x/can_y said {self.can_xy} but {override} says {found}; "
                    f"using the file, which is what the simulator was actually given")
            self.can_xy = found
        self.data.qpos[self.can_qadr:self.can_qadr + 3] = [
            self.can_xy[0], self.can_xy[1], table_z + self.can_half]
        self.data.qpos[self.can_qadr + 3:self.can_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.carried_by = None
        self.grasp_offset = np.zeros(3)

        self.tray_xy = (float(self.get_parameter("tray_x").value),
                        float(self.get_parameter("tray_y").value))
        self.tray_top = float(self.get_parameter("tray_top_z").value)
        self.tray_r = float(self.get_parameter("tray_radius").value)

        self.pubs = {c: self.create_publisher(Image, f"{self.ns}/{c}/image_raw", 10)
                     for c in self.cams}
        self.state = None
        self.create_subscription(Float64MultiArray, "/zero/eef_state", self._on_state, 10)
        self.frames = 0
        self.held = 0
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info(
            f"shadow: rendering {len(self.cams)} cameras from {path} to {self.ns}/*  "
            f"can starts at {self.can_xy}")

    def _on_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == 20:
            self.state = np.asarray(msg.data, dtype=float)

    def _ik(self, side: str, pos: np.ndarray, rot: np.ndarray) -> float:
        """Drive the shadow arm onto a pose. Returns the position residual in metres."""
        a = self.arm[side]
        for _ in range(60):
            mujoco.mj_forward(self.model, self.data)
            err = pos - self.data.site_xpos[a["site"]]
            cur = self.data.site_xmat[a["site"]].reshape(3, 3)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (cur.T @ rot).flatten())
            n = np.linalg.norm(q[1:])
            ang = 2.0 * np.arctan2(n, q[0])
            if np.linalg.norm(err) < 5e-4 and abs(ang) < 5e-3:
                break
            rerr = cur @ (q[1:] / (n + 1e-12) * ang)
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jp, jr, a["site"])
            jac = np.vstack([jp[:, a["dof"]], jr[:, a["dof"]]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.05 ** 2 * np.eye(6),
                                         np.concatenate([err, 0.6 * rerr]))
            for k, j in enumerate(a["jid"]):
                v = self.data.qpos[a["qadr"][k]] + float(np.clip(dq[k], -0.05, 0.05))
                if self.model.jnt_limited[j]:
                    v = float(np.clip(v, *self.model.jnt_range[j]))
                self.data.qpos[a["qadr"][k]] = v
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(pos - self.data.site_xpos[a["site"]]))

    @staticmethod
    def _can_from_override(path: str, target_mjcf: str):
        """Read the can's xy from the <key qpos=...> the launch handed the simulator.

        Indexes the can's free joint using the TARGET model's layout, so the value is exact rather
        than inferred. Returns None if either file is missing, leaving can_x/can_y in force.
        """
        if not target_mjcf:
            return None
        try:
            import re
            mo = re.search(r'qpos="([^"]+)"', open(path).read())
            if mo is None:
                return None
            q = np.asarray([float(v) for v in mo.group(1).split()])
            tm = mujoco.MjModel.from_xml_path(target_mjcf)
        except (OSError, ValueError, RuntimeError):
            return None
        bid = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_BODY, "obj_root")
        if bid < 0:
            return None
        jid = next((j for j in range(tm.njnt)
                    if tm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and tm.jnt_bodyid[j] == bid), -1)
        if jid < 0 or tm.nq != len(q):
            return None
        adr = tm.jnt_qposadr[jid]
        return (float(q[adr]), float(q[adr + 1]))

    def _rest_on_surface(self) -> None:
        """Sit the can down on whatever is under it.

        This node never steps physics, it only does forward kinematics, so a released can would
        otherwise hang in mid-air exactly where it was let go.
        """
        pos = self.data.qpos[self.can_qadr:self.can_qadr + 3].copy()
        if np.linalg.norm(pos[:2] - np.asarray(self.tray_xy)) < self.tray_r:
            top, where = self.tray_top, "the tray"
        else:
            top, where = self.table_z, "the table"
        pos[2] = top + self.can_half
        self.data.qpos[self.can_qadr:self.can_qadr + 3] = pos
        self.data.qpos[self.can_qadr + 3:self.can_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        return where

    def _place_can(self, poses: dict, grips: dict) -> None:
        """Track the can: sitting, carried by one hand, or handed from one hand to the other."""
        can = self.data.qpos[self.can_qadr:self.can_qadr + 3].copy()
        if self.carried_by is None:
            for side in SIDES:
                if grips[side] < self.grip_closed:
                    tool = self.data.site_xpos[self.arm[side]["site"]].copy()
                    if np.linalg.norm(tool[:2] - can[:2]) < 0.08:
                        self.carried_by = side
                        self.grasp_offset = can - tool
                        self.get_logger().info(f"shadow: can picked up by the {side} hand")
                        break
        elif grips[self.carried_by] >= self.grip_closed:
            # The giving hand has opened. In a handover the other hand is already closed around the
            # can, so the can goes with it rather than falling; that is the whole point of the task.
            gave = self.carried_by
            other = "right" if gave == "left" else "left"
            tool_o = self.data.site_xpos[self.arm[other]["site"]].copy()
            if grips[other] < self.grip_closed and np.linalg.norm(tool_o - can) < 0.12:
                self.carried_by = other
                self.grasp_offset = can - tool_o
                self.get_logger().info(
                    f"shadow: can handed from the {gave} hand to the {other} hand")
            else:
                self.carried_by = None
                where = self._rest_on_surface()
                self.get_logger().info(
                    f"shadow: can released by the {gave} hand with no other hand on it, "
                    f"setting it down on {where}")
        if self.carried_by is not None:
            tool = self.data.site_xpos[self.arm[self.carried_by]["site"]]
            self.data.qpos[self.can_qadr:self.can_qadr + 3] = tool + self.grasp_offset

    def _tick(self) -> None:
        if self.state is None:
            return
        poses, grips = unpack(self.state)
        worst = 0.0
        for side in SIDES:
            pos, rot = poses[side]
            worst = max(worst, self._ik(side, pos, rot))
        if worst > self.max_res:
            self.held += 1
            if self.held % 20 == 1:
                self.get_logger().warn(
                    f"shadow arm is {worst * 1000:.0f} mm from the pose the target is in; "
                    f"holding the last good frame rather than publishing a wrong one")
            return
        self.held = 0
        self._place_can(poses, grips)
        mujoco.mj_forward(self.model, self.data)
        now = self.get_clock().now().to_msg()
        for cam in self.cams:
            self.renderer.update_scene(self.data, camera=cam)
            rgb = self.renderer.render()
            msg = Image()
            msg.header.stamp = now
            msg.header.frame_id = f"{cam}_optical_frame"
            msg.height, msg.width = rgb.shape[0], rgb.shape[1]
            msg.encoding = "rgb8"
            msg.is_bigendian = 0
            msg.step = 3 * msg.width
            msg.data = rgb.tobytes()
            self.pubs[cam].publish(msg)
        self.frames += 1
        if self.frames % 100 == 0:
            self.get_logger().info(f"shadow: {self.frames} frames, worst IK {worst * 1000:.1f} mm")


def main() -> None:
    rclpy.init()
    node = ShadowRender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
