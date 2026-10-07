"""The 14-dim tool-frame force observation, shared by the recorder and the policy.

This exists so there is exactly one implementation. The recorder writes this vector into the
dataset and the policy reads it at inference, and if the two ever computed it differently the model
would be fed a quantity it was never trained on, silently and with no error anywhere. Two copies of
this arithmetic in two nodes is that bug waiting to happen.

Per hand, 7 numbers:

    net wrench (6)  the sum of the finger wrenches, rotated into the tool frame. This is the
                    external load: the object's weight, or the gripper pressing on something. On a
                    symmetric pinch it is near zero because the pads cancel.
    squeeze (1)     the mean of the per-finger force magnitudes, divided by this robot's own
                    grip-force cap. Grip strength, which the sum destroys: gripping the can reads
                    15.6 N and 14.5 N on the two pads, so the sum is ~1 N and the mean is ~15 N.

Sum and mean are both defined for any number of fingers, so a two-finger jaw and a three-finger
hand present the same 7 numbers. Normalising by the robot's own cap is what makes squeeze
comparable across embodiments, and is why the cap must come from the generated config rather than
being hard-coded.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import WrenchStamped

SIDES = ("left", "right")


def assemble(sensors, side, rot, grip_force, readings, bias: float = 0.0) -> np.ndarray:
    """The arithmetic, as a pure function. Both callers route through here.

    sensors: names, side: "left"/"right" per sensor, rot: [n,3,3] sensor->tool,
    grip_force: this robot's cap in N, readings: {name: 6-vector}.

    bias is the no-load reading subtracted before normalising, default 0 so a gripper without
    linkage preload is unaffected. It exists because squeeze has to mean the same thing on every
    embodiment: in the training data the channel is sharply bimodal, ~0.045 when empty and ~1.0
    when holding, with only 1.3% of frames between 0.15 and 0.35. A Robotiq 2F-85's four-bar
    carries 2.4 N with the jaws open and nothing held, which lands its idle squeeze at 0.24 --
    inside that empty band, so the policy reads "partly gripping" whenever the hand is in fact
    open. Subtracting the no-load reading makes the channel measure grip rather than preload.
    """
    out = []
    for s in SIDES:
        w = np.zeros(6)
        mags = []
        for i, sensor in enumerate(sensors):
            if side[i] != s:
                continue
            r = rot[i]
            v = readings[sensor]
            f = r @ v[:3]
            w[:3] += f
            w[3:] += r @ v[3:]
            mags.append(float(np.linalg.norm(f)))
        squeeze = (max(sum(mags) / len(mags) - bias, 0.0) / grip_force) if mags else 0.0
        out.append(np.concatenate([w, [squeeze]]))
    return np.concatenate(out).astype(np.float32)


class ToolFrameForce:
    """Subscribes to the fingertip wrench broadcasters and assembles the 14-dim vector.

    Reads the same `/**` parameters the recorder does, so both nodes are configured identically by
    construction. Call `declare(node)` before `node` has read its parameters, then `vector()` per
    tick.
    """

    PARAMS = ("ft_sensors", "ft_rot", "ft_side", "grip_force", "ft_bias")

    def __init__(self, node) -> None:
        self.node = node
        self.sensors = [s for s in node.get_parameter("ft_sensors").value if s]
        self.side = [x for x in node.get_parameter("ft_side").value if x]
        self.grip_force = max(float(node.get_parameter("grip_force").value), 1e-6)
        self.bias = float(node.get_parameter("ft_bias").value)

        rot = np.asarray(node.get_parameter("ft_rot").value, dtype=float)
        if self.sensors:
            if rot.size != 9 * len(self.sensors) or len(self.side) != len(self.sensors):
                raise ValueError(
                    f"ft_rot/ft_side do not match {len(self.sensors)} sensors "
                    f"({rot.size} rotation values, {len(self.side)} sides). Regenerate the config "
                    f"with scripts/gen_bringup.py.")
            self.rot = rot.reshape(len(self.sensors), 3, 3)
        else:
            self.rot = np.zeros((0, 3, 3))

        self.readings: dict[str, np.ndarray] = {s: np.zeros(6) for s in self.sensors}
        for s in self.sensors:
            node.create_subscription(
                WrenchStamped, f"/{s}_broadcaster/wrench",
                lambda m, s=s: self._on_wrench(m, s), 10)

    @staticmethod
    def declare(node) -> None:
        node.declare_parameter("ft_sensors", [""])
        node.declare_parameter("ft_rot", [0.0])
        node.declare_parameter("ft_side", [""])
        node.declare_parameter("grip_force", 1.0)
        node.declare_parameter("ft_bias", 0.0)

    def _on_wrench(self, msg: WrenchStamped, sensor: str) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        self.readings[sensor] = np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=float)

    def vector(self) -> np.ndarray:
        return assemble(self.sensors, self.side, self.rot, self.grip_force, self.readings,
                        self.bias)
