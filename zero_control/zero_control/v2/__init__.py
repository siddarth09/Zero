"""ZERO v2 model additions, layered on SmolVLA rather than forking it.

Three changes, each with its own module so they can be enabled and measured one at a time:

    noise.py    correlated noise sampled through the Cholesky factor of the action chunk's own
                temporal covariance, instead of independent per-step Gaussian noise
    vision.py   a frozen DINOv2 token stream beside SigLIP
    policy.py   joins them to SmolVLA, and adds force as an input

Overlapping chunks, the other thing we wanted, comes from LeRobot's own RTC rather than a second
implementation here. Depth was planned and dropped: no depth column in the 82-episode set.
"""
