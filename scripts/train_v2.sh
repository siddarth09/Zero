#!/usr/bin/env bash
# ZERO v2 training: SmolVLA plus correlated noise, DINOv2 and force.
#
#   bash scripts/train_v2.sh [STEPS] [RUN_NAME] [DATASET_ROOT]
#
# The dataset defaults to crossv2_vlfa, NOT crossv2_base. crossv2_base is the stripped training
# view and has no force column, so training on it silently disables the F in VLFA: the key is
# simply absent from the batch and the force projection never receives anything.
set -eo pipefail

STEPS="${1:-130000}"
RUN="${2:-zerov2_c50}"
DATA="${3:-$HOME/zero_data/crossv2_vlfa}"
CHUNK="${CHUNK:-50}"
NOISE="${NOISE:-$HOME/zero_data/crossv2_vlfa/noise_cholesky_c50.pt}"
DINO_POOL="${DINO_POOL:-2}"     # 2 -> 64 tokens/camera. 1 is PRANA v3's 256, and 4x the KV cost.

if [ ! -f "$NOISE" ]; then
  echo "ERROR: no noise factor at $NOISE" >&2
  echo "  fit one first, at the SAME chunk size:" >&2
  echo "  /home/sid/lerobot_env/bin/python scripts/fit_noise.py $DATA $CHUNK 0.9 $NOISE" >&2
  exit 1
fi

echo "free space on \$HOME: $(df -h "$HOME" | awk 'NR==2{print $4}')"
echo "dataset $DATA   chunk $CHUNK   dino_pool $DINO_POOL"

/home/sid/lerobot_env/bin/python scripts/train_v2.py \
  --dataset.repo_id=zero/crossv2 \
  --dataset.root="$DATA" \
  --policy.type=zerovla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.resize_imgs_with_padding="[256,256]" \
  --policy.chunk_size="$CHUNK" \
  --policy.n_action_steps="$CHUNK" \
  --policy.noise_factor_path="$NOISE" \
  --policy.use_correlated_noise=true \
  --policy.use_dino=true \
  --policy.dino_pool="$DINO_POOL" \
  --policy.use_force=true \
  --policy.scheduler_decay_steps="$STEPS" \
  --batch_size=8 \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir="$HOME/zero_runs/$RUN"
