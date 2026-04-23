from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from src.utils.batch import BatchedData
from einops import rearrange
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AENet(torch.nn.Module):
    def __init__(
        self,
        model_name,
        max_batch_size,
        dinov2_model=None,
        dinov2_repo_dir="./dinov2",
        patch_size=14,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.dinov2_repo_dir = dinov2_repo_dir
        self.dinov2_model = dinov2_model or self._load_dinov2_model()
        self.max_batch_size = max_batch_size
        self.patch_size = patch_size
        logger.info("Initialize AENet done!")

    def _load_dinov2_model(self):
        repo_dir = Path(to_absolute_path(str(self.dinov2_repo_dir))).resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv2 repo not found at {repo_dir}. Set user.dinov2_repo_dir "
                "or model.ae_net.dinov2_repo_dir to your local DINOv2 checkout."
            )
        return torch.hub.load(str(repo_dir), self.model_name, source="local")

    def compute_features(self, images):
        # with torch.no_grad():  # no gradients
        features = self.dinov2_model.forward_features(images)
        return features

    def reshape_local_features(self, local_features, num_patches):
        local_features = rearrange(
            local_features, "b (h w) c -> b c h w", h=num_patches[0], w=num_patches[1]
        )
        return local_features

    def forward_by_chunk(self, processed_rgbs, patch_dim=[2, 3]):
        batch_rgbs = BatchedData(batch_size=self.max_batch_size, data=processed_rgbs)
        patch_features = BatchedData(batch_size=self.max_batch_size)

        num_patche_h = processed_rgbs.shape[patch_dim[0]] // self.patch_size
        num_patche_w = processed_rgbs.shape[patch_dim[1]] // self.patch_size

        for idx_sample in range(len(batch_rgbs)):
            feats = self.compute_features(batch_rgbs[idx_sample])
            patch_feats = self.reshape_local_features(
                feats["x_prenorm"][:, 1:, :],
                num_patches=[num_patche_h, num_patche_w],
            )
            patch_features.cat(patch_feats)
        return F.normalize(patch_features.data, dim=1)

    def forward(self, images):
        features = self.forward_by_chunk(images)
        return features
