"""ZERO v2: SmolVLA plus correlated noise, DINOv2 and force.

Three additions, layered onto the stock policy rather than forking it, so upstream fixes still
arrive and each piece can be switched off to isolate what it bought:

    correlated noise  flow matching starts from a draw shaped like a real trajectory instead of
                      white noise, so single samples are smooth rather than smooth-on-average
    DINOv2            frozen ViT-S/14 patch tokens beside SigLIP, carrying geometry that a
                      language-contrastive encoder drops
    force             the 6-D tool-frame wrench and normalised squeeze per hand, the F in VLFA

Depth was planned and dropped: the 82-episode set has no depth column, so it would have trained on
nothing. See vision.py.

Rolling chunk overlap, the fourth thing we wanted, is NOT here: LeRobot already ships it as RTC
(Real Time Chunking), which solves the same problem with a scheduled guidance weight rather than a
hard mask. Turn it on through `rtc_config` instead of reimplementing it.

Why the model is patched rather than subclassed: SmolVLAPolicy.__init__ constructs VLAFlowMatching
by name, so a subclass of the model would still require rebuilding the policy and loading the
500M-parameter VLM a second time. Attaching to the instance keeps one load and one source of truth.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field

import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from zero_control.v2.noise import CorrelatedNoiseSampler
from zero_control.v2.vision import CAMERA_ORDER, V2VisionTokens

FORCE_KEY = "observation.force"


@PreTrainedConfig.register_subclass("zerovla")
@dataclass
class ZeroVLAConfig(SmolVLAConfig):
    """SmolVLA's config plus the v2 switches. Defaults reproduce v1 apart from the noise."""

    use_correlated_noise: bool = True
    noise_factor_path: str = ""          # the .pt written by scripts/fit_noise.py
    use_dino: bool = True
    dino_pool: int = 1                   # 1 keeps PRANA v3's 256 tokens/camera, 2 pools to 64
    use_force: bool = True
    force_dim: int = 14
    camera_order: tuple[str, ...] = field(default_factory=lambda: CAMERA_ORDER)


def _sample_noise_correlated(self, shape, device):
    """Replaces VLAFlowMatching.sample_noise. Falls back to white noise when unfitted."""
    return self.noise_sampler.sample(shape, device)


def _embed_prefix_v2(self, images, img_masks, lang_tokens, lang_masks, state=None):
    """SmolVLA's prefix, with the v2 tokens appended.

    They go after the state token with mask_ar 0, which puts them in the state's attention block:
    they see all of vision and language, and state sees them. Appending rather than splicing keeps
    this a small override, so an upstream change to embed_prefix does not silently break it.
    """
    embs, pad_masks, att_masks = self._embed_prefix_base(
        images, img_masks, lang_tokens, lang_masks, state)

    extra = []
    if getattr(self, "v2_vision", None) is not None:
        tok = self.v2_vision(images=self._v2_images)
        if tok is not None:
            extra.append(tok)
    if getattr(self, "force_proj", None) is not None and self._v2_force is not None:
        extra.append(self.force_proj(self._v2_force)[:, None, :])
    if not extra:
        return embs, pad_masks, att_masks

    extra = torch.cat(extra, dim=1).to(embs.dtype)
    b, n = extra.shape[0], extra.shape[1]
    embs = torch.cat([embs, extra], dim=1)
    pad_masks = torch.cat(
        [pad_masks, torch.ones(b, n, dtype=pad_masks.dtype, device=pad_masks.device)], dim=1)
    att_masks = torch.cat(
        [att_masks, torch.zeros(b, n, dtype=att_masks.dtype, device=att_masks.device)], dim=1)
    return embs, pad_masks, att_masks


def attach_v2(policy: SmolVLAPolicy, config: ZeroVLAConfig) -> SmolVLAPolicy:
    """Wire the v2 additions onto a constructed SmolVLA policy."""
    model = policy.model
    hidden = model.state_proj.out_features

    if config.use_correlated_noise:
        sampler = CorrelatedNoiseSampler(config.max_action_dim, config.chunk_size)
        if config.noise_factor_path:
            blob = torch.load(config.noise_factor_path, weights_only=False, map_location="cpu")
            if tuple(blob["cholesky_L"].shape) != tuple(sampler.cholesky_L.shape):
                raise ValueError(
                    f"noise factor is {tuple(blob['cholesky_L'].shape)} but this config needs "
                    f"{tuple(sampler.cholesky_L.shape)}; refit with chunk_size={config.chunk_size}")
            sampler.cholesky_L.copy_(blob["cholesky_L"])
            sampler.fitted.fill_(True)
        model.noise_sampler = sampler
        model.sample_noise = types.MethodType(_sample_noise_correlated, model)

    model.v2_vision = V2VisionTokens(
        hidden_dim=hidden, camera_order=tuple(config.camera_order),
        use_dino=config.use_dino, dino_pool=config.dino_pool) if config.use_dino else None

    model.force_proj = nn.Linear(config.force_dim, hidden) if config.use_force else None

    # The batch never reaches embed_prefix, so stash what it needs per forward pass.
    model._v2_images = None
    model._v2_force = None
    if not hasattr(model, "_embed_prefix_base"):
        model._embed_prefix_base = model.embed_prefix
        model.embed_prefix = types.MethodType(_embed_prefix_v2, model)

    _wrap_batch_capture(policy, config)
    return policy


def _wrap_batch_capture(policy: SmolVLAPolicy, config: ZeroVLAConfig) -> None:
    """Pull the v2 inputs out of the batch before SmolVLA drops them.

    SmolVLA only forwards the keys it knows about, so force would never reach the model
    otherwise. Wrapping the three public entry points keeps this in one place.
    """
    order = tuple(config.camera_order)
    logged = {"done": False}

    def latest(t: torch.Tensor, rank: int) -> torch.Tensor:
        """Drop a leading observation-history axis if there is one.

        The dataset yields force as [B, D] and images as [B, C, H, W], but LeRobot's preprocessing
        can add an observation-step axis before the policy sees them. Everything downstream wants a
        single timestep, so take the most recent one rather than assume a rank.
        """
        while t.ndim > rank:
            t = t[:, -1]
        return t

    def capture(batch):
        model = policy.model
        imgs = {c: latest(batch[k], 4) for c in order
                for k in (f"observation.images.{c}",) if k in batch}
        model._v2_images = imgs if len(imgs) == len(order) else None
        f = batch.get(FORCE_KEY) if config.use_force else None
        model._v2_force = latest(f, 2) if f is not None else None

        if not logged["done"]:
            logged["done"] = True
            got = {k: tuple(v.shape) for k, v in (imgs or {}).items()}
            print(f"[zerov2] first batch: images {got} "
                  f"force {None if model._v2_force is None else tuple(model._v2_force.shape)}",
                  flush=True)

    for name in ("forward", "select_action", "predict_action_chunk"):
        original = getattr(policy, name)

        def wrapped(batch, *a, _orig=original, **kw):
            capture(batch)
            return _orig(batch, *a, **kw)

        setattr(policy, name, wrapped)


class ZeroVLAPolicy(SmolVLAPolicy):
    """SmolVLA with the v2 additions attached at construction."""

    config_class = ZeroVLAConfig
    name = "zerovla"

    def __init__(self, config: ZeroVLAConfig, **kwargs) -> None:
        # Upstream takes (config, **kwargs) and handles normalisation through the processor
        # pipeline, not a dataset_stats argument. Pass kwargs straight through so this keeps
        # working if that signature grows.
        super().__init__(config, **kwargs)
        attach_v2(self, config)
