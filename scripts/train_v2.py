"""Training entry point for ZERO v2.

    $HOME/lerobot_env/bin/python scripts/train_v2.py --policy.type=zerovla ...
"""

from __future__ import annotations

import sys
from pathlib import Path

# The v2 package lives in the ROS package so the robot can import it at inference too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zero_control"))

import lerobot.policies.factory as factory  # noqa: E402
from zero_control.v2.policy import ZeroVLAPolicy  # noqa: E402,F401  (registers "zerovla")

_original_get_policy_class = factory.get_policy_class


def _get_policy_class(name: str):
    if name == "zerovla":
        return ZeroVLAPolicy
    return _original_get_policy_class(name)


factory.get_policy_class = _get_policy_class


def main() -> None:
    from lerobot.scripts.lerobot_train import main as lerobot_main
    lerobot_main()


if __name__ == "__main__":
    main()
