"""Extra visual token stream for ZERO v2: frozen DINOv2.

SmolVLA sees the world through SigLIP, inside SmolVLM2. SigLIP is trained against language, so it
is strong on what a thing is and comparatively weak on where it is. Every failure this project hit
in transfer was metric rather than semantic: a lens 15 mm too close, a 90 mm standoff against 45, a
57 mm object offset, a 39 mm hover that never resolved. So add a stream that carries geometry: frozen DINOv2 ViT-S/14. Self-supervised, and its patch
features keep spatial structure that language-contrastive features drop. Same encoder PRANA v3
uses, and pairing it with SigLIP is the DINOv2+SigLIP fusion Prismatic VLMs established and OpenVLA
inherits.

Depth was the obvious companion and is deliberately absent: the 82-episode training set has no
depth column. Only cross_v1 (51 episodes) and rebot_pick_place recorded it, so a depth stream here
would have been dead weight that trains on nothing. Bring it back when there is data for it.

Tokens come out at the action expert's width, tagged with which camera they came from, to be
concatenated onto SmolVLA's own prefix. Nothing here replaces SigLIP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# DINOv2 is patch-14, so the input side must be a multiple of 14. 224 gives a 16x16 grid, i.e. the
# 256 tokens per camera PRANA v3 uses. SmolVLA feeds SigLIP at 512; the two are resized separately.
DINO_SIDE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Camera identity is a learned embedding indexed by position, so this order IS the meaning of those
# indices. It must be identical at training and at inference or every camera embedding is wrong.
# Note it deliberately differs from the order used elsewhere in ZERO (front, left_wrist,
# right_wrist): the manipulating hands come first here. Because they differ, callers pass a dict
# keyed by name rather than a pre-stacked tensor, and this module does the ordering.
CAMERA_ORDER = ("left_wrist", "right_wrist", "front")


class DinoV2Stream(nn.Module):
    """Frozen DINOv2 patch tokens, one bank per camera, projected to the expert width."""

    def __init__(self, hidden_dim: int, num_cameras: int, pool: int = 1,
                 model_name: str = "dinov2_vits14") -> None:
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name,
                                       trust_repo=True, verbose=False)
        self.embed_dim = int(self.backbone.embed_dim)
        # Frozen means frozen: no grad, and eval() pinned in train() below so BN/dropout inside the
        # backbone cannot drift with the rest of the model.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.pool = int(pool)
        self.camera_embed = nn.Embedding(num_cameras, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, hidden_dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        # The frozen trunk IS saved in the checkpoint, deliberately. Excluding it looks free (22.1M
        # unchanging parameters, ~90 MB a checkpoint, and 175 lines of expected-missing keys in the
        # load log) and is not: filtering state_dict() while load_state_dict() still sees the real
        # parameters desyncs the two, and safetensors.load_model asserts that its own view of
        # unexpected keys matches torch's. That assertion fires on resume and kills the run. If the
        # log noise ever needs fixing, fix it at the log, not by lying about the state dict.

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        self.backbone.eval()          # never let the frozen trunk enter train mode
        return self

    def tokens_per_camera(self) -> int:
        g = DINO_SIDE // 14
        return (g // self.pool) ** 2

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)["x_norm_patchtokens"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, K, 3, H, W] in [0, 1]. Returns [B, K * tokens_per_camera, hidden_dim]."""
        b, k = images.shape[:2]
        x = images.flatten(0, 1)
        if x.shape[-1] != DINO_SIDE or x.shape[-2] != DINO_SIDE:
            x = F.interpolate(x, size=(DINO_SIDE, DINO_SIDE), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self._features(x)                                    # [B*K, 256, 384]

        if self.pool > 1:
            g = DINO_SIDE // 14
            feats = feats.transpose(1, 2).reshape(b * k, self.embed_dim, g, g)
            feats = F.avg_pool2d(feats, self.pool)
            feats = feats.flatten(2).transpose(1, 2)

        # Which camera a patch came from is not recoverable from the patch itself once the banks are
        # concatenated, and it matters: the same red cylinder means different things in the giving
        # wrist and the receiving wrist.
        cam = self.camera_embed(torch.arange(k, device=images.device))
        feats = feats + cam.repeat_interleave(b, 0).unsqueeze(1)
        return self.proj(feats).reshape(b, -1, self.proj.out_features)


class V2VisionTokens(nn.Module):
    """The DINOv2 stream, ready to concatenate onto SmolVLA's prefix."""

    def __init__(self, hidden_dim: int = 960, camera_order: tuple[str, ...] = CAMERA_ORDER,
                 use_dino: bool = True, dino_pool: int = 1) -> None:
        super().__init__()
        self.camera_order = tuple(camera_order)
        self.dino = DinoV2Stream(hidden_dim, len(self.camera_order), dino_pool) if use_dino else None

    def camera_index(self, name: str) -> int:
        return self.camera_order.index(name)

    def stack(self, by_name: dict[str, torch.Tensor]) -> torch.Tensor:
        """Order a {camera name: [B, C, H, W]} dict into [B, K, C, H, W] by CAMERA_ORDER.

        Missing cameras are an error rather than a silent drop: quietly shrinking K would shift
        every later camera's embedding index and mislabel the whole batch.
        """
        missing = [c for c in self.camera_order if c not in by_name]
        if missing:
            raise KeyError(f"missing camera(s) {missing}; expected all of {self.camera_order}")
        return torch.stack([by_name[c] for c in self.camera_order], dim=1)

    def n_tokens(self, num_cameras: int | None = None) -> int:
        num_cameras = len(self.camera_order) if num_cameras is None else num_cameras
        return 0 if self.dino is None else num_cameras * self.dino.tokens_per_camera()

    def forward(self, images: torch.Tensor | dict | None = None) -> torch.Tensor | None:
        """Accepts either a name-keyed dict (preferred) or a tensor already in CAMERA_ORDER."""
        if self.dino is None or images is None:
            return None
        if isinstance(images, dict):
            images = self.stack(images)
        return self.dino(images)
