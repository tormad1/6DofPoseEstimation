import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.dataloader.test import build_frame_batch
from src.utils.dataset import LMO_ID_to_index
from src.utils.logging import get_logger
import src.utils.tensor_collection as tc

logger = get_logger(__name__)


class GigaPoseRuntime:
    """Small inference wrapper intended to be imported by scripts or embedded CPython.

    The CLI still uses Hydra to build the model and datasets, but callers that
    already have a runtime can pass one RGB frame plus detections directly via
    ``run_frame`` or ``run_rgb_bytes``.
    """

    def __init__(
        self,
        model,
        transforms,
        dataset_name,
        device="cpu",
        run_id="cpu_rgb",
        output_dir=None,
    ):
        self.model = model
        self.transforms = transforms
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir is not None else None

        self.model.test_dataset_name = dataset_name
        self.model.run_id = run_id
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_config_for_frames(
        cls,
        cfg: DictConfig,
        base_dir=None,
    ) -> "GigaPoseRuntime":
        OmegaConf.set_struct(cfg, False)
        normalize_config_paths(cfg, base_dir=base_dir)
        configure_cpu_threads(OmegaConf.select(cfg, "machine.cpu_threads"))

        device = torch.device("cpu")
        transforms = instantiate(cfg.data.transform)
        model = instantiate(cfg.model)
        load_model_checkpoint(model, cfg.model.checkpoint_path, device)
        model.checkpoint_cache_key = make_file_cache_key(cfg.model.checkpoint_path)

        template_cfg = copy.deepcopy(cfg.data.test.dataloader)
        template_cfg.dataset_name = cfg.test_dataset_name
        template_cfg.batch_size = cfg.machine.batch_size
        template_cfg.test_setting = cfg.test_setting
        template_cfg._target_ = "src.dataloader.template.TemplateSet"
        template_dataset = instantiate(template_cfg)

        model.template_datasets = {cfg.test_dataset_name: template_dataset}
        model.max_num_dets_per_forward = cfg.max_num_dets_per_forward

        runtime = cls(
            model=model,
            transforms=transforms,
            dataset_name=cfg.test_dataset_name,
            device=device,
            run_id=cfg.run_id or "cpu_rgb",
            output_dir=cfg.save_dir,
        )
        return runtime

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        base_dir=None,
    ) -> tuple["GigaPoseRuntime", object]:
        runtime = cls.from_config_for_frames(cfg, base_dir=base_dir)

        dataloader_cfg = cfg.data.test.dataloader
        dataloader_cfg.dataset_name = cfg.test_dataset_name
        dataloader_cfg.batch_size = cfg.machine.batch_size
        dataloader_cfg.test_setting = cfg.test_setting
        test_dataset = instantiate(dataloader_cfg)
        runtime.transforms = test_dataset.transforms
        return runtime, test_dataset

    def make_dataloader(self, dataset, num_workers=0):
        return DataLoader(
            dataset.scene_dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )

    def run_dataloader(self, dataloader, save_predictions=True):
        results = []
        with torch.inference_mode():
            for batch_id, batch in enumerate(dataloader):
                predictions = self.run_batch(
                    batch=batch,
                    batch_id=batch_id,
                    save_predictions=save_predictions,
                )
                results.extend(self.predictions_to_dicts(predictions))

        if save_predictions:
            self.model.on_test_epoch_end()
        return results

    def warmup(self):
        """Precompute or load template features before the first video frame."""
        self.model.warmup_templates(self.dataset_name)
        return self

    def run_frame(
        self,
        rgb,
        K,
        detections,
        scene_id=1,
        im_id=1,
        save_predictions=False,
        batch_id=0,
    ):
        """Run one RGB frame.

        Args:
            rgb: uint8 image shaped ``H x W x 3`` in RGB channel order.
            K: 3x3 camera intrinsics matrix.
            detections: List of dictionaries. Each detection needs ``bbox`` in
                xywh pixel format and ``category_id`` or ``obj_id``. ``score``,
                ``time``, ``mask`` and COCO ``segmentation`` are optional.
            scene_id: Metadata id used in returned records.
            im_id: Metadata id used in returned records.
            save_predictions: Whether to save the intermediate npz batch file.
            batch_id: Required to build the save path when saving.
        """
        detections = list(detections or [])
        if not detections:
            return []

        batch = build_frame_batch(
            rgb=rgb,
            K=K,
            detections=detections,
            transforms=self.transforms,
            dataset_name=self.dataset_name,
            scene_id=scene_id,
            im_id=im_id,
        )
        predictions = self.run_batch(
            batch=batch,
            batch_id=batch_id,
            save_predictions=save_predictions,
        )
        return self.predictions_to_dicts(predictions)

    def run_rgb_bytes(
        self,
        rgb_bytes,
        width,
        height,
        K,
        detections,
        scene_id=1,
        im_id=1,
        channels=3,
        channel_order="RGB",
        stride=None,
        save_predictions=False,
        batch_id=0,
    ):
        """Run one frame from a native byte buffer.

        ``rgb_bytes`` can be ``bytes``, ``bytearray`` or any Python buffer object
        exposed by an embedding layer. Pixels are expected to be uint8. Set
        ``channel_order`` to ``BGR``, ``RGBA`` or ``BGRA`` when the native buffer
        is not already RGB. If rows are padded, pass ``stride`` in bytes.
        """
        rgb = rgb_bytes_to_array(
            rgb_bytes=rgb_bytes,
            width=width,
            height=height,
            channels=channels,
            channel_order=channel_order,
            stride=stride,
        )
        return self.run_frame(
            rgb=rgb,
            K=K,
            detections=detections,
            scene_id=scene_id,
            im_id=im_id,
            save_predictions=save_predictions,
            batch_id=batch_id,
        )

    def run_roi(
        self,
        roi_rgb,
        K,
        bbox_xywh,
        object_id=1,
        score=1.0,
        mask=None,
        scene_id=1,
        im_id=1,
        save_predictions=False,
        batch_id=0,
    ):
        """Run one cropped ROI while preserving full-frame camera geometry.

        ``bbox_xywh`` is the ROI rectangle in the original full camera image.
        If ``roi_rgb`` has been resized after cropping, intrinsics are scaled to
        the ROI image size before inference.
        """
        roi_rgb = np.asarray(roi_rgb, dtype=np.uint8)
        if roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3:
            raise ValueError("roi_rgb must be a uint8 HxWx3 RGB array")

        roi_h, roi_w = roi_rgb.shape[:2]
        K_roi = adjust_intrinsics_for_roi(
            K=K,
            bbox_xywh=bbox_xywh,
            roi_width=roi_w,
            roi_height=roi_h,
        )
        batch = build_roi_batch(
            rgb=roi_rgb,
            K=K_roi,
            object_id=object_id,
            score=score,
            mask=mask,
            transforms=self.transforms,
            dataset_name=self.dataset_name,
            scene_id=scene_id,
            im_id=im_id,
        )
        predictions = self.run_batch(
            batch=batch,
            batch_id=batch_id,
            save_predictions=save_predictions,
        )
        poses = self.predictions_to_dicts(predictions)
        return poses[0] if poses else None

    def run_roi_bytes(
        self,
        roi_bytes,
        width,
        height,
        K,
        bbox_xywh,
        object_id=1,
        score=1.0,
        mask=None,
        scene_id=1,
        im_id=1,
        channels=3,
        channel_order="RGB",
        stride=None,
        save_predictions=False,
        batch_id=0,
    ):
        """Run one cropped ROI from a native byte buffer."""
        roi_rgb = rgb_bytes_to_array(
            rgb_bytes=roi_bytes,
            width=width,
            height=height,
            channels=channels,
            channel_order=channel_order,
            stride=stride,
        )
        return self.run_roi(
            roi_rgb=roi_rgb,
            K=K,
            bbox_xywh=bbox_xywh,
            object_id=object_id,
            score=score,
            mask=mask,
            scene_id=scene_id,
            im_id=im_id,
            save_predictions=save_predictions,
            batch_id=batch_id,
        )

    def run_batch(self, batch, batch_id=None, save_predictions=False):
        batch = batch.to(self.device)
        with torch.inference_mode():
            predictions, total_time = self.model.predict_batch(
                batch=batch,
                dataset_name=self.dataset_name,
            )
            _, predictions = self.model.attach_test_metadata(
                predictions=predictions,
                test_list=batch.test_list,
                time=total_time,
            )

        if save_predictions:
            if batch_id is None:
                raise ValueError("batch_id is required when save_predictions=True")
            save_path = Path(self.model.log_dir) / "predictions" / f"{batch_id}.npz"
            self.model.save_batch_predictions(predictions, save_path)
        return predictions

    @staticmethod
    def predictions_to_dicts(predictions, top_k=1):
        poses = predictions.pred_poses.detach().cpu().numpy()
        scores = predictions.scores.detach().cpu().numpy()
        scene_ids = np.asarray(predictions.infos.scene_id).astype(np.int32)
        im_ids = np.asarray(predictions.infos.view_id).astype(np.int32)
        object_ids = np.asarray(predictions.infos.label).astype(np.int32)

        if poses.ndim == 3:
            poses = poses[:, None]
        if scores.ndim == 1:
            scores = scores[:, None]

        results = []
        num_hypotheses = min(top_k, poses.shape[1])
        for idx in range(poses.shape[0]):
            for hyp_id in range(num_hypotheses):
                pose = poses[idx, hyp_id]
                results.append(
                    {
                        "scene_id": int(scene_ids[idx]),
                        "im_id": int(im_ids[idx]),
                        "object_id": int(object_ids[idx]),
                        "hypothesis_id": int(hyp_id),
                        "score": float(scores[idx, hyp_id]),
                        "R": pose[:3, :3].tolist(),
                        "t": pose[:3, 3].tolist(),
                    }
                )
        return results


def create_runtime(config_dir=None, config_name="test", overrides=None):
    """Create a frame-only runtime from config without invoking the CLI."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = Path(config_dir) if config_dir is not None else repo_root / "configs"
    if not config_dir.is_absolute():
        config_dir = repo_root / config_dir
    config_dir = config_dir.resolve()

    if config_name.endswith(".yaml"):
        config_name = Path(config_name).stem

    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        global_hydra.clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    return GigaPoseRuntime.from_config_for_frames(cfg, base_dir=repo_root)


def create_runtime_from_paths(
    checkpoint_path=None,
    dinov2_repo_dir=None,
    dataset_root_dir=None,
    template_dir=None,
    dataset_name="T282",
    output_dir=None,
    template_cache_dir=None,
    run_id="embedded",
    num_templates=162,
    max_num_dets_per_forward=1,
    cpu_threads=None,
    warmup=False,
):
    """Create the embedded ROI runtime without Hydra composition."""
    from torchvision.transforms import Compose, Normalize

    from src.dataloader.template import TemplateSet
    from src.models.gigaPose import GigaPose
    from src.models.matching import LocalSimilarity
    from src.models.network.ae_net import AENet
    from src.models.network.ist_net import ISTNet, Regressor
    from src.models.network.resnet import ResNet
    from src.utils.crop import CropResizePad

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = resolve_runtime_path(
        checkpoint_path or repo_root / "gigaPose_datasets/pretrained/gigaPose_v1.ckpt",
        repo_root,
    )
    dinov2_repo_dir = resolve_runtime_path(
        dinov2_repo_dir or repo_root / "dinov2",
        repo_root,
    )
    dataset_root_dir = Path(
        resolve_runtime_path(
            dataset_root_dir or repo_root / "gigaPose_datasets/datasets",
            repo_root,
        )
    )
    template_dir = resolve_runtime_path(
        template_dir or dataset_root_dir / "templates",
        repo_root,
    )
    output_dir = resolve_runtime_path(
        output_dir or repo_root / "gigaPose_datasets/results/embedded",
        repo_root,
    )
    template_cache_dir = resolve_runtime_path(
        template_cache_dir or repo_root / "gigaPose_datasets/template_cache",
        repo_root,
    )

    configure_cpu_threads(cpu_threads)
    transforms = SimpleNamespace(
        normalize=Compose(
            [
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )
            ]
        ),
        crop_transform=CropResizePad(target_size=224),
    )
    ae_net = AENet(
        model_name="dinov2_vitl14",
        max_batch_size=64,
        dinov2_repo_dir=dinov2_repo_dir,
    )
    backbone = ResNet(
        config={
            "n_heads": 0,
            "input_dim": 3,
            "input_size": 256,
            "initial_dim": 128,
            "block_dims": [128, 192, 256, 512],
            "descriptor_size": 256,
        }
    )
    regressor = Regressor(
        descriptor_size=256,
        hidden_dim=256,
        use_tanh_act=True,
        normalize_output=True,
    )
    ist_net = ISTNet(
        model_name="resnet",
        backbone=backbone,
        regressor=regressor,
        max_batch_size=64,
    )
    model = GigaPose(
        model_name="large",
        ae_net=ae_net,
        ist_net=ist_net,
        testing_metric=LocalSimilarity(k=1, sim_threshold=0.5, patch_threshold=3),
        log_dir=output_dir,
        max_num_dets_per_forward=max_num_dets_per_forward,
        template_cache_dir=template_cache_dir,
        template_cache_enabled=True,
    )
    load_model_checkpoint(model, checkpoint_path, torch.device("cpu"))
    model.checkpoint_cache_key = make_file_cache_key(checkpoint_path)

    template_cfg = SimpleNamespace(
        dir=template_dir,
        scale_factor=1.0,
        num_templates=int(num_templates),
        pose_name="object_poses/OBJECT_ID.npy",
    )
    template_dataset = TemplateSet(
        root_dir=dataset_root_dir,
        dataset_name=dataset_name,
        template_config=template_cfg,
        transforms=transforms,
    )
    model.template_datasets = {dataset_name: template_dataset}

    runtime = GigaPoseRuntime(
        model=model,
        transforms=transforms,
        dataset_name=dataset_name,
        device="cpu",
        run_id=run_id,
        output_dir=output_dir,
    )
    if warmup:
        runtime.warmup()
    return runtime


def normalize_config_paths(cfg: DictConfig, base_dir=None):
    cfg.machine.root_dir = resolve_runtime_path(cfg.machine.root_dir, base_dir)
    cfg.save_dir = resolve_runtime_path(cfg.save_dir, base_dir)
    cfg.model.checkpoint_path = resolve_runtime_path(
        cfg.model.checkpoint_path,
        base_dir,
    )
    if OmegaConf.select(cfg, "model.template_cache_dir") is not None:
        cfg.model.template_cache_dir = resolve_runtime_path(
            cfg.model.template_cache_dir,
            base_dir,
        )
    cfg.data.test.dataloader.root_dir = resolve_runtime_path(
        cfg.data.test.dataloader.root_dir,
        base_dir,
    )
    cfg.data.test.dataloader.template_config.dir = resolve_runtime_path(
        cfg.data.test.dataloader.template_config.dir,
        base_dir,
    )


def resolve_runtime_path(path_value, base_dir=None):
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    if base_dir is not None:
        return str((Path(base_dir) / path).resolve())
    return to_absolute_path(str(path))


def configure_cpu_threads(num_threads=None, interop_threads=None):
    if num_threads is not None:
        num_threads = int(num_threads)
        if num_threads > 0:
            torch.set_num_threads(num_threads)
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
    if interop_threads is not None:
        interop_threads = int(interop_threads)
        if interop_threads > 0:
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError as exc:
                logger.warning(f"Could not set interop threads: {exc}")
    return {
        "num_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }


def build_roi_batch(
    rgb,
    K,
    object_id,
    transforms,
    dataset_name="custom",
    score=1.0,
    mask=None,
    scene_id=1,
    im_id=1,
):
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a uint8 HxWx3 array")

    height, width = rgb.shape[:2]
    model_object_id = int(object_id)
    if "lmo" in dataset_name:
        model_object_id = LMO_ID_to_index[model_object_id]

    rgb_tensor = torch.as_tensor(rgb).permute(2, 0, 1).float() / 255.0
    if mask is None:
        mask_tensor = torch.ones((height, width), dtype=torch.float32)
    else:
        mask_tensor = torch.as_tensor(np.asarray(mask).astype(np.float32))
        if tuple(mask_tensor.shape) != (height, width):
            raise ValueError("ROI mask must match roi_rgb height and width")

    masked_rgba = torch.cat(
        [rgb_tensor * mask_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)],
        dim=0,
    ).unsqueeze(0)
    full_roi_box = torch.tensor([[0, 0, width, height]], dtype=torch.long)
    cropped_data = transforms.crop_transform(full_roi_box, images=masked_rgba)

    infos = pd.DataFrame(
        [
            {
                "label": str(model_object_id),
                "scene_id": str(scene_id),
                "view_id": str(im_id),
                "visib_fract": float(score),
            }
        ]
    )
    test_list = tc.PandasTensorCollection(
        infos=pd.DataFrame(
            [
                {
                    "im_id": int(im_id),
                    "scene_id": int(scene_id),
                    "obj_id": int(model_object_id),
                    "inst_count": 1,
                    "detection_time": 0.0,
                }
            ]
        )
    )
    return tc.PandasTensorCollection(
        tar_img=transforms.normalize(cropped_data["images"][:, :3]),
        tar_mask=cropped_data["images"][:, -1],
        tar_K=torch.as_tensor(np.asarray(K, dtype=np.float32)).reshape(1, 3, 3),
        tar_M=cropped_data["M"],
        infos=infos,
        test_list=test_list,
    )


def adjust_intrinsics_for_roi(K, bbox_xywh, roi_width=None, roi_height=None):
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    x, y, width, height = np.asarray(bbox_xywh, dtype=np.float32).reshape(4)
    if width <= 0 or height <= 0:
        raise ValueError("bbox_xywh width and height must be positive")

    roi_width = width if roi_width is None else float(roi_width)
    roi_height = height if roi_height is None else float(roi_height)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("roi_width and roi_height must be positive")

    scale_x = roi_width / width
    scale_y = roi_height / height
    full_to_roi = np.array(
        [
            [scale_x, 0.0, -scale_x * x],
            [0.0, scale_y, -scale_y * y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(full_to_roi @ K)


def make_file_cache_key(path):
    path = Path(path)
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def load_model_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing checkpoint keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected checkpoint keys: {len(unexpected)}")


def rgb_bytes_to_array(
    rgb_bytes,
    width,
    height,
    channels=3,
    channel_order="RGB",
    stride=None,
):
    width = int(width)
    height = int(height)
    channels = int(channels)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if channels not in (3, 4):
        raise ValueError("channels must be 3 or 4")
    channel_order = str(channel_order).upper()
    valid_channels = {"R", "G", "B", "A"}
    if len(channel_order) != channels or not set(channel_order).issubset(valid_channels):
        raise ValueError(
            "channel_order must describe the source channels, "
            "e.g. RGB, BGR, RGBA or BGRA"
        )
    for required_channel in ("R", "G", "B"):
        if required_channel not in channel_order:
            raise ValueError("channel_order must include R, G and B")

    row_size = width * channels
    stride = row_size if stride is None else int(stride)
    if stride < row_size:
        raise ValueError("stride must be at least width * channels")

    expected_size = stride * height
    buffer = memoryview(rgb_bytes)
    if buffer.nbytes < expected_size:
        raise ValueError(
            f"rgb buffer is too small: got {buffer.nbytes} bytes, "
            f"need at least {expected_size}"
        )

    flat = np.frombuffer(buffer, dtype=np.uint8, count=expected_size)
    rows = flat.reshape(height, stride)[:, :row_size]
    image = rows.reshape(height, width, channels)
    rgb_indexes = [channel_order.index(channel) for channel in "RGB"]
    return np.ascontiguousarray(image[:, :, rgb_indexes])
