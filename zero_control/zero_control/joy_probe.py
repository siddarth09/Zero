"""Print which gamepad control you just moved, so the teleop map can be verified not assumed.

    ros2 run zero_control joy_probe
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

# Standard XInput layout, used only to label what moved.
AXES = {0: "left stick X (left +)", 1: "left stick Y (up +)", 2: "LT trigger",
        3: "right stick X (left +)", 4: "right stick Y (up +)", 5: "RT trigger",
        6: "D-pad X", 7: "D-pad Y"}
BUTTONS = {0: "A", 1: "B", 2: "X", 3: "Y", 4: "LB", 5: "RB", 6: "BACK", 7: "START",
           8: "GUIDE", 9: "left stick click", 10: "right stick click"}


class JoyProbe(Node):
    def __init__(self) -> None:
        super().__init__("zero_joy_probe")
        self.rest: list[float] | None = None
        self.prev_b: list[int] = []
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.get_logger().info(
            "move one stick / trigger / button at a time; Ctrl-C when done")

    def _on_joy(self, msg: Joy) -> None:
        if self.rest is None:
            self.rest = list(msg.axes)
            print(f"resting axes: {[round(a, 3) for a in self.rest]}")
            print(f"{len(msg.axes)} axes, {len(msg.buttons)} buttons\n")
            self.prev_b = list(msg.buttons)
            return
        for i, v in enumerate(msg.axes):
            if abs(v - self.rest[i]) > 0.25:
                bar = int(abs(v) * 20)
                print(f"  axis {i:2d}  {AXES.get(i, '?'):26} {v:+6.3f}  "
                      f"{'-' if v < 0 else '+'}{'#' * bar}")
        for i, v in enumerate(msg.buttons):
            was = self.prev_b[i] if i < len(self.prev_b) else 0
            if v and not was:
                print(f"  button {i:2d} {BUTTONS.get(i, '?')}")
        self.prev_b = list(msg.buttons)


def main() -> None:
    rclpy.init()
    node = JoyProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
