#!/usr/bin/env bash
# Play a trained checkpoint closed-loop in the sim.
#
#   1) start the sim:  ros2 launch zero_bringup rebot.launch.py can_x:=0.47 can_y:=0.24
#      ...or the target:  ros2 launch zero_bringup vx300s.launch.py can_x:=0.47 can_y:=0.24
#                         then run this with  ROBOT=vx300s
#   2) then, in another terminal:  bash scripts/run_policy.sh
#   3) it starts on its own; Ctrl-C to stop
#
# Set JOY=1 to bring back the gamepad instead, so X starts/stops the policy the way the recorder
# does. Without it no joystick is needed and none is launched.
#
# Args: $1 = checkpoint dir, $2 = replan horizon in steps (default 25). Anything after those is
# forwarded to the node verbatim, so any policy_node parameter can be set from here, e.g.
#     bash scripts/run_policy.sh "" 25 -p block_until_settled:=false
# which matters because the settle wait costs up to settle_timeout_s PER ACTION STEP: with the
# residual never reaching settle_m the 10 Hz loop degrades to ~1 Hz, and the policy then sees
# observations ten times staler than anything in its training data.
#
# `set -u` is deliberately not used: the ROS setup scripts reference unbound variables.
set -eo pipefail

# xargs trims whitespace: a trailing space after a line-continuation backslash makes the shell
# pass a single-space argument, which `${1:-default}` treats as set. That reached rcl as
# `-p checkpoint:=" "`, which it parses as unset, and the node died with
# "parameter 'checkpoint' is not initialized".
CKPT="$(printf '%s' "${1:-}" | xargs || true)"
CKPT="${CKPT:-$HOME/zero_runs/crossv2_full_c25/checkpoints/last/pretrained_model}"
STEPS="$(printf '%s' "${2:-}" | xargs || true)"
STEPS="${STEPS:-25}"
# Forward any remaining arguments to the node untouched.
[ $# -ge 1 ] && shift || true
[ $# -ge 1 ] && shift || true

# Which robot to drive. The policy is embodiment-agnostic: it emits the 20-dim absolute EEF pose
# from action.py, and each robot's own eef_control_node turns that into joint commands through
# its own IK. Only the params file differs; nothing about the checkpoint changes here.
ROBOT="${ROBOT:-rebot}"

# The dataset the checkpoint was TRAINED on, read out of the checkpoint rather than hardcoded.
# It supplies the normalisation stats, and the wrong ones do not error: STATE and ACTION are
# MEAN_STD, so mismatched stats silently rescale every action and the arm moves plausibly but
# wrongly, which is indistinguishable from a failed transfer. v1 trained on zero/cross and v2 on
# zero/crossv2, and this script used to name v1's unconditionally.
if [ -f "$CKPT/train_config.json" ]; then
  read -r _rid _root <<<"$(python3 -c "
import json,sys
c=json.load(open('$CKPT/train_config.json'))['dataset']
print(c.get('repo_id',''), c.get('root',''))
" 2>/dev/null)"
  REPO_ID="${REPO_ID:-$_rid}"
  DATASET_ROOT="${DATASET_ROOT:-$_root}"
fi
REPO_ID="${REPO_ID:-zero/cross}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/zero_data/cross_base}"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint dir not found: '$CKPT'" >&2
  echo "  usage: bash scripts/run_policy.sh [CHECKPOINT_DIR] [N_ACTION_STEPS]" >&2
  echo "  available:" >&2
  ls -d "$HOME"/zero_runs/*/checkpoints/*/pretrained_model 2>/dev/null | sed 's/^/    /' >&2
  exit 1
fi
case "$STEPS" in (''|*[!0-9]*) echo "ERROR: N_ACTION_STEPS must be an integer, got '$STEPS'" >&2; exit 1;; esac
WS=/home/sid/projects25

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

# The policy needs lerobot_env (torch cu128 for sm_120, lerobot 0.5.1 for PEFT loading). That venv
# is built with include-system-site-packages=false, so ROS's site-packages must be added by hand.
# zero_control comes from source because the install tree holds only an egg-link, which the venv's
# interpreter does not process. That also means node edits apply without a colcon build.
export PYTHONPATH="$WS/src/ZERO/zero_control:${PYTHONPATH:-}"

# No gamepad by default: policy_node's `autostart` sets running=True at construction, and its
# /joy subscription simply never fires when nothing publishes, so the joystick is not needed to
# run a policy. Ctrl-C stops it.
#
# JOY=1 restores the button. The X button comes from /joy, published by the `joy` driver, not by
# zero_control's teleop node. rebot_teleop.launch.py starts both, so using that launch here would
# leave teleop publishing /zero/eef_target at 50 Hz and holding the home pose, which overrides
# every policy command (the policy asked for 162 mm and the arm moved 2.3 mm). So start joy_node
# alone, and only if nothing is already publishing /joy. autorepeat_rate matches the teleop
# launch: joy_node otherwise publishes only on change, so a button press can be missed.
if [ -n "${JOY:-}" ]; then
  AUTOSTART=false
  if ! ros2 topic info /joy 2>/dev/null | grep -q "Publisher count: [1-9]"; then
    echo "starting joy_node (nothing is publishing /joy); press X to start the policy"
    ros2 run joy joy_node --ros-args \
      -p deadzone:=0.05 -p autorepeat_rate:=50.0 -p coalesce_interval_ms:=5 &
    JOY_PID=$!
    # Not a trap plus exec: `exec` replaces this shell, so the trap is discarded and Ctrl-C never
    # runs it, which leaked five joy_node processes over one debugging session. Run the node as a
    # child instead and clean up after it returns.
    sleep 2
  else
    echo "/joy already has a publisher, reusing it; press X to start the policy"
  fi
else
  AUTOSTART=true
  echo "no gamepad: the policy starts immediately (set JOY=1 to require X instead)"
fi

# Anything publishing /zero/eef_target will fight the policy for the arm; the node itself also
# refuses to start in that case, but say so here where the fix is obvious.
if ros2 node list 2>/dev/null | grep -q '^/zero_teleop$'; then
  echo
  echo "ERROR: zero_teleop is running and publishes /zero/eef_target at 50 Hz, which will"
  echo "       override every policy command. Stop it first:"
  echo "         pkill -f 'zero_control/teleop'"
  exit 1
fi

cleanup() { [ -n "${JOY_PID:-}" ] && kill "$JOY_PID" 2>/dev/null; true; }
trap cleanup EXIT INT TERM

/home/sid/lerobot_env/bin/python -m zero_control.policy_node --ros-args \
  --params-file "$WS/install/zero_bringup/share/zero_bringup/config/${ROBOT}_control.yaml" \
  -p checkpoint:="$CKPT" \
  -p repo_id:="$REPO_ID" \
  -p dataset_root:="$DATASET_ROOT" \
  -p n_action_steps:="$STEPS" \
  -p autostart:="$AUTOSTART" \
  ${SHADOW:+-p camera_ns:="/shadow"} \
  ${TRACE:+-p trace_path:="$TRACE"} \
  ${DUMP:+-p frame_dump_dir:="$DUMP"} \
  ${DUMP_EVERY:+-p frame_dump_every:="$DUMP_EVERY"} \
  ${DUMP_MAX:+-p frame_dump_max:="$DUMP_MAX"} \
  "$@"
