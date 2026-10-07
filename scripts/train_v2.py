"""Training entry point for ZERO v2.

    /home/sid/lerobot_env/bin/python scripts/train_v2.py --policy.type=zerovla ...

This is a thin shim, not a training loop. LeRobot's `lerobot-train` resolves a policy type through
`factory.get_policy_class`, which is a hardcoded if/elif chain and therefore cannot see a policy
defined outside the package. Rather than fork the trainer or vendor a copy of its loop (checkpoint
resume, schedulers, accelerate, dataloader workers, all of which already work), this teaches the
factory one new name and then hands control straight back.

Two details make this safe:

  * `factory.make_policy` calls `get_policy_class(...)` as a module global, so replacing the module
    attribute takes effect at call time. Patching the imported name inside another module would not.
  * `ZeroVLAConfig` registers itself as "zerovla" on import, which is what lets the CLI parse
    `--policy.type=zerovla` at all. So the import below is load-bearing, not decorative.

Weights come from `--policy.pretrained_path=lerobot/smolvla_base`. The v2 modules (DINOv2
projection, force projection, noise factor) are absent from that checkpoint;
PreTrainedPolicy loads with `strict=False` and logs the missing keys, so they keep their
initialisation while everything SmolVLA pretrained is restored. Read that log: if keys you expected
to load appear as missing, the run is starting colder than you think.
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
