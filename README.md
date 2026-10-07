# ZERO

Cross-embodiment manipulation transfer. Demonstrations are teleoperated on one arm and the
resulting policy drives a different arm without retraining, by expressing actions and state only
as end-effector poses in the table frame — `pos(3) + rot6d(6) + grip(1)` per hand.

Embodiments: Seeed reBot, Trossen ViperX 300s, UR5e, Unitree G1.

## Requirements

ROS 2 Jazzy, `mujoco_ros2_control`, `mujoco`, `pinocchio`. Recording and training additionally
need `lerobot` and `rerun-sdk` in a separate venv (`$LEROBOT_ENV`, default `~/lerobot_env`).

## Build

```bash
cd ~/projects25
colcon build --symlink-install --packages-select zero_description zero_bringup zero_control
source install/setup.bash
```

`ros2` commands run from the workspace root. Scripts run from `src/ZERO`.

## View the sim

```bash
cd ~/projects25/src/ZERO
python3 scripts/view_scene.py rebot        # or vx300s, ur5e, g1
```

## Record demonstrations

Three terminals. Swap `rebot` for `vx300s`, `ur5e` or `g1`.

```bash
# 1. simulator + controllers + IK
ros2 launch zero_bringup rebot.launch.py
ros2 launch zero_bringup rebot.launch.py can_x:=0.30 can_y:=0.42 can_yaw:=0.5

# 2. gamepad teleoperation
ros2 launch zero_bringup rebot_teleop.launch.py

# 3. recorder
TASK="pick up the red cylinder hand it over to the robot on the right and place it on the black tray"
ros2 run zero_control record --ros-args \
    --params-file install/zero_bringup/share/zero_bringup/config/rebot_control.yaml \
    -p root:=$HOME/zero_data/cross_v1 \
    -p task:="$TASK"
```

Gamepad: `X` start/stop an episode, `B` discard and re-record, `LB`/`RB` select arm, `Y` toggle
gripper. Keyboard: `SPACE` start, `RIGHT` save, `LEFT` re-record, `ESC` exit.

Point the recorder at an existing root to append to it.

## Train

```bash
cd ~/projects25/src/ZERO
python3 scripts/make_train_view.py ~/zero_data/cross_v2 crossv2
bash scripts/train_base_full.sh 30000 crossv2_full_c25 $HOME/zero_data/crossv2_base
```

`train_base_lora.sh` is the LoRA variant, `train_v2.sh` trains the v2 policy.

## Run a policy

```bash
pkill -f 'zero_control/teleop'
ros2 launch zero_bringup rebot.launch.py can_x:=0.48 can_y:=0.52
TRACE=$HOME/rollout.npz bash scripts/run_policy.sh
```

`X` on the gamepad starts and stops the rollout. Pass a checkpoint and action-step count as
positional arguments to override the defaults:

```bash
bash scripts/run_policy.sh outputs/train/crossv2_full_c25/checkpoints/last/pretrained_model 10
```

Dump the observations the policy acted on:

```bash
DUMP=/tmp/run DUMP_EVERY=25 DUMP_MAX=120 ROBOT=vx300s bash scripts/run_policy.sh "" 10
```

## Regenerate robot descriptions

```bash
cd ~/projects25/src/ZERO
bash scripts/fetch_vendor.sh
python3 scripts/gen_urdf.py rebot
python3 scripts/gen_scene.py rebot
python3 scripts/gen_bringup.py rebot
```

## Layout

```
zero_bringup/     launch files and per-embodiment controller config
zero_control/     ROS 2 nodes: teleop, eef_control, record, policy, shadow_render
zero_description/ generated URDF, MJCF and meshes
robots/           source MJCF models
scripts/          generators, dataset tools, training and rollout entry points
```
