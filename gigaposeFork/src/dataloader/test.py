from __future__ import annotations

import json
from typing import List

# Third Party
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# MegaPose
import src.utils.tensor_collection as tc
from src.custom_megapose.transform import Transform
from src.dataloader.scene import (
    CameraData,
    ObjectData,
    ObservationInfos,
    SceneObservation,
)
from src.utils.bbox import BoundingBox
from src.utils.logging import get_logger
from src.utils.inout import load_test_list_and_cnos_detections
from src.utils.dataset import LMO_ID_to_index

logger = get_logger(__name__)


class BOPSceneDataset(Dataset):
    def __init__(self, root_dir: Path, dataset_name: str, image_keys: List[str]):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.scene_root = self.root_dir / self.dataset_name / "test_scenewise"
        self.image_keys = sorted(image_keys, key=_image_key_sort_key)
        self._camera_cache = {}

        if not self.scene_root.exists():
            raise FileNotFoundError(f"Missing scene directory: {self.scene_root}")

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, index):
        image_key = self.image_keys[index]
        scene_id, im_id = [int(part) for part in image_key.split("_")]
        scene_dir = self.scene_root / f"{scene_id:06d}"

        rgb_path = self._find_rgb_path(scene_dir, im_id)
        rgb = np.array(Image.open(rgb_path).convert("RGB"))

        camera = self._load_camera(scene_dir, im_id)
        if "cam_R_w2c" in camera:
            R = np.asarray(camera["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
            t = np.asarray(camera["cam_t_w2c"], dtype=np.float32).reshape(3)
        else:
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)

        return SceneObservation(
            rgb=rgb,
            infos=ObservationInfos(scene_id=str(scene_id), view_id=str(im_id)),
            object_datas=[],
            camera_data=CameraData(
                K=np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3),
                TWC=Transform(R, t),
                resolution=rgb.shape[:2],
            ),
            binary_masks={},
        )

    def _load_camera(self, scene_dir: Path, im_id: int):
        scene_id = scene_dir.name
        if scene_id not in self._camera_cache:
            camera_path = scene_dir / "scene_camera.json"
            self._camera_cache[scene_id] = json.loads(camera_path.read_text())

        cameras = self._camera_cache[scene_id]
        return cameras.get(str(im_id)) or cameras[f"{im_id:06d}"]

    def _find_rgb_path(self, scene_dir: Path, im_id: int) -> Path:
        for image_dir in ("rgb", "gray"):
            for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                path = scene_dir / image_dir / f"{im_id:06d}{suffix}"
                if path.exists():
                    return path
        raise FileNotFoundError(f"Missing RGB image for scene={scene_dir.name} im={im_id}")


class FrameBatchBuilder:
    def __init__(self, dataset_name, transforms, test_list, cnos_dets):
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.test_list = test_list
        self.cnos_dets = cnos_dets

    def load_test_list(self, batch: List[SceneObservation]):
        target_lists = []
        detection_times = []
        for scene_obs in batch:
            infos = scene_obs.infos
            scene_id, im_id = int(infos.scene_id), int(infos.view_id)
            image_key = f"{scene_id:06d}_{im_id:06d}"
            target_list = self.test_list[image_key]
            target_lists.append(target_list)
            if image_key in self.cnos_dets:
                detection_times.append(self.cnos_dets[image_key][0]["time"])
            else:
                logger.warning(f"Image key {image_key} not in cnos_dets!")
                detection_times.append(0)
        assert len(target_lists) == len(detection_times)

        target_list_dict = {
            "im_id": [],
            "scene_id": [],
            "obj_id": [],
            "inst_count": [],
            "detection_time": [],
        }
        for idx, target_list in enumerate(target_lists):
            if len(target_list) == 0:
                continue
            for target in target_list:
                for name in target_list_dict:
                    if name == "detection_time":
                        continue
                    if name == "obj_id" and "lmo" in self.dataset_name:
                        target_list_dict[name].append(LMO_ID_to_index[target[name]])
                    else:
                        target_list_dict[name].append(target[name])
                target_list_dict["detection_time"].append(detection_times[idx])
        target_list_pd = pd.DataFrame(target_list_dict)
        return tc.PandasTensorCollection(infos=target_list_pd)

    def add_detections(self, batch: List[SceneObservation]):
        for scene_obs in batch:
            infos = scene_obs.infos
            scene_id, im_id = int(infos.scene_id), int(infos.view_id)
            image_key = f"{scene_id:06d}_{im_id:06d}"
            dets = self.cnos_dets[image_key]
            object_datas = []
            binary_masks = {}
            for idx, det in enumerate(dets):
                data = {}
                data["visib_fract"] = det.get("score", 1.0)
                data["bbox_modal"] = det["bbox"]
                data["bbox_amodal"] = det["bbox"]
                data["label"] = str(det.get("category_id", det.get("obj_id")))
                data["unique_id"] = f"{idx+1}"
                data["TWO"] = [[0, 0, 0, 1], [0, 0, 0]]
                object_data = ObjectData.from_json(data)
                object_datas.append(object_data)

                binary_masks[object_data.unique_id] = detection_to_mask(
                    det, scene_obs.rgb.shape[:2]
                )
            scene_obs.object_datas = object_datas
            scene_obs.binary_masks = binary_masks
        return batch

    def double_check_test_list(self, real_data, test_list):
        labels = np.asarray(real_data.infos.label).astype(np.int32)
        test_obj_ids = test_list.infos.obj_id
        assert np.allclose(np.unique(labels), np.unique(test_obj_ids))

    def process_real(self, batch):
        rgb = batch["rgb"] / 255.0
        detections = batch["gt_detections"]
        data = batch["gt_data"]

        bboxes = BoundingBox(detections.bboxes, "xywh")
        idx_selected = np.arange(len(detections.bboxes))
        bboxes = bboxes.reset(idx_selected)

        batch_im_id = detections[idx_selected].infos.batch_im_id
        masks = data.masks[idx_selected]
        K = data.K[idx_selected].float()
        rgb = rgb[batch_im_id]
        masked_rgba = torch.cat([rgb * masks[:, None, :, :], masks[:, None, :, :]], dim=1)
        cropped_data = self.transforms.crop_transform(bboxes.xyxy_box, images=masked_rgba)

        return tc.PandasTensorCollection(
            K=K,
            rgb=cropped_data["images"][:, :3],
            mask=cropped_data["images"][:, -1],
            M=cropped_data["M"],
            infos=data[idx_selected].infos,
        )

    def collate_fn(
        self,
        batch: List[SceneObservation],
    ):
        test_list = self.load_test_list(batch)
        batch = self.add_detections(batch)
        batch = SceneObservation.collate_fn(batch)
        real_data = self.process_real(batch)

        if "lmo" in self.dataset_name:
            new_labels = real_data.infos.label
            real_data.infos.label = [str(LMO_ID_to_index[int(label)]) for label in new_labels]

        out_data = tc.PandasTensorCollection(
            tar_img=self.transforms.normalize(real_data.rgb),
            tar_mask=real_data.mask,
            tar_K=real_data.K,
            tar_M=real_data.M,
            infos=real_data.infos,
            test_list=test_list,
        )
        self.double_check_test_list(real_data, test_list)
        return out_data


class GigaPoseTestSet:
    def __init__(
        self,
        batch_size,
        root_dir,
        dataset_name,
        template_config,
        transforms,
        test_setting,
    ):
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.transforms = transforms

        # depending on setting:
        # 1. localization: load target_objects
        # 2. detection: not target_objects
        assert test_setting in [
            "localization",
            "detection",
        ], f"{test_setting} not supported!"
        self.load_detections(test_setting=test_setting)
        self.scene_dataset = BOPSceneDataset(
            root_dir=self.root_dir,
            dataset_name=self.dataset_name,
            image_keys=list(self.test_list.keys()),
        )
        self.batch_builder = FrameBatchBuilder(
            dataset_name=self.dataset_name,
            transforms=self.transforms,
            test_list=self.test_list,
            cnos_dets=self.cnos_dets,
        )

    def load_detections(self, test_setting):
        if test_setting == "localization":
            max_det_per_object_id = 32 if self.dataset_name == "icbin" else 16
        else:
            max_det_per_object_id = None
        self.test_list, self.cnos_dets = load_test_list_and_cnos_detections(
            self.root_dir,
            self.dataset_name,
            test_setting,
            max_det_per_object_id=max_det_per_object_id,
        )

    def collate_fn(
        self,
        batch: List[SceneObservation],
    ):
        return self.batch_builder.collate_fn(batch)


def decode_binary_mask(segmentation):
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for COCO RLE segmentation masks. "
            "Pass a binary mask or bbox-only detection to avoid this dependency."
        ) from exc

    rle = dict(segmentation)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    return coco_mask.decode(rle).astype(bool)


def detection_to_mask(detection, image_shape):
    x, y, w, h = [int(round(v)) for v in detection["bbox"]]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, image_shape[1]), min(y + h, image_shape[0])

    if "mask" in detection:
        mask = np.asarray(detection["mask"]).astype(bool)
        if mask.ndim != 2:
            raise ValueError("detection mask must be a 2D array")
        if tuple(mask.shape) == tuple(image_shape):
            return mask

        full_mask = np.zeros(image_shape, dtype=bool)
        roi_h = max(y1 - y0, 0)
        roi_w = max(x1 - x0, 0)
        mask_roi = mask[:roi_h, :roi_w]
        full_mask[
            y0 : y0 + mask_roi.shape[0],
            x0 : x0 + mask_roi.shape[1],
        ] = mask_roi
        return full_mask

    if "segmentation" in detection:
        return decode_binary_mask(detection["segmentation"])

    mask = np.zeros(image_shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def make_scene_observation(rgb, K, scene_id=1, im_id=1):
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a uint8 HxWx3 array")

    return SceneObservation(
        rgb=rgb,
        infos=ObservationInfos(scene_id=str(scene_id), view_id=str(im_id)),
        object_datas=[],
        camera_data=CameraData(
            K=np.asarray(K, dtype=np.float32).reshape(3, 3),
            TWC=Transform(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
            resolution=rgb.shape[:2],
        ),
        binary_masks={},
    )


def make_test_list_from_detections(detections, scene_id=1, im_id=1):
    counts = {}
    for detection in detections:
        obj_id = int(detection.get("category_id", detection.get("obj_id")))
        counts[obj_id] = counts.get(obj_id, 0) + 1

    return [
        {
            "scene_id": int(scene_id),
            "im_id": int(im_id),
            "obj_id": obj_id,
            "inst_count": inst_count,
        }
        for obj_id, inst_count in counts.items()
    ]


def normalize_detections(detections):
    """Validate and normalize ROI detections for direct frame inference.

    Each detection needs:
      - ``bbox``: xywh pixels, with positive width and height.
      - ``category_id`` or ``obj_id``: object id used by the templates.

    Optional keys are ``score``, ``time``, ``mask`` and COCO ``segmentation``.
    The returned dictionaries always include ``category_id``, ``score`` and
    ``time``.
    """
    normalized = []
    if detections is None:
        return normalized

    for detection in detections:
        detection = dict(detection)
        if "bbox" not in detection:
            raise ValueError("each detection must include bbox in xywh format")
        bbox = np.asarray(detection["bbox"], dtype=np.float32).reshape(-1)
        if bbox.shape[0] != 4:
            raise ValueError("detection bbox must contain four values: x, y, w, h")
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("detection bbox width and height must be positive")
        detection["bbox"] = [float(value) for value in bbox]

        if "category_id" not in detection and "obj_id" in detection:
            detection["category_id"] = detection["obj_id"]
        if "category_id" not in detection:
            raise ValueError("each detection must include category_id or obj_id")
        detection["category_id"] = int(detection["category_id"])

        if "score" not in detection:
            detection["score"] = 1.0
        detection["score"] = float(detection["score"])
        if "time" not in detection:
            detection["time"] = 0.0
        detection["time"] = float(detection["time"])
        normalized.append(detection)
    return normalized


def build_frame_batch(
    rgb,
    K,
    detections,
    transforms,
    dataset_name="custom",
    scene_id=1,
    im_id=1,
):
    """Build the tensor batch used by GigaPose from one RGB frame and detections."""
    image_key = f"{int(scene_id):06d}_{int(im_id):06d}"
    detections = normalize_detections(detections)
    builder = FrameBatchBuilder(
        dataset_name=dataset_name,
        transforms=transforms,
        test_list={image_key: make_test_list_from_detections(detections, scene_id, im_id)},
        cnos_dets={image_key: detections},
    )
    observation = make_scene_observation(rgb, K, scene_id=scene_id, im_id=im_id)
    return builder.collate_fn([observation])


def _image_key_sort_key(image_key):
    scene_id, im_id = image_key.split("_")
    return int(scene_id), int(im_id)
