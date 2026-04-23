from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import src.utils.tensor_collection as tc
from src.custom_megapose.transform import Transform
from src.utils.tensor_collection import PandasTensorCollection

Resolution = Tuple[int, int]
SceneObservationTensorCollection = PandasTensorCollection


@dataclass
class ObjectData:
    label: str
    TWO: Optional[Transform] = None
    unique_id: Optional[int] = None
    bbox_amodal: Optional[np.ndarray] = None
    bbox_modal: Optional[np.ndarray] = None
    visib_fract: Optional[float] = None
    TWO_init: Optional[Transform] = None

    @staticmethod
    def from_json(data: Dict) -> "ObjectData":
        label = data["label"]
        assert isinstance(label, str)
        obj = ObjectData(label=label)

        for key in ("TWO", "TWO_init"):
            if key in data:
                rotation, translation = data[key]
                obj.__dict__[key] = Transform(tuple(rotation), tuple(translation))

        for key in ("unique_id", "visib_fract"):
            if key in data:
                obj.__dict__[key] = data[key]

        for key in ("bbox_amodal", "bbox_modal"):
            if key in data:
                obj.__dict__[key] = np.asarray(data[key])
        return obj


@dataclass
class CameraData:
    K: Optional[np.ndarray] = None
    resolution: Optional[Resolution] = None
    TWC: Optional[Transform] = None
    camera_id: Optional[str] = None
    TWC_init: Optional[Transform] = None


@dataclass
class ObservationInfos:
    scene_id: str
    view_id: str


@dataclass
class SceneObservation:
    rgb: Optional[np.ndarray] = None
    infos: Optional[ObservationInfos] = None
    object_datas: Optional[List[ObjectData]] = None
    camera_data: Optional[CameraData] = None
    binary_masks: Optional[Dict[int, np.ndarray]] = None

    @staticmethod
    def collate_fn(
        batch: List["SceneObservation"], object_labels: Optional[List[str]] = None
    ) -> Dict[Any, Any]:
        if object_labels is not None:
            object_labels = set(object_labels)

        cam_infos, K = [], []
        im_infos = []
        gt_data = []
        gt_detections = []
        initial_data = []
        rgb_images = []

        for batch_im_id, data in enumerate(batch):
            im_infos.append(
                {
                    "scene_id": data.infos.scene_id,
                    "view_id": data.infos.view_id,
                    "batch_im_id": batch_im_id,
                }
            )
            K.append(data.camera_data.K)
            cam_infos.append(
                {
                    "TWC": data.camera_data.TWC,
                    "resolution": data.camera_data.resolution,
                }
            )
            rgb_images.append(torch.as_tensor(data.rgb).permute(2, 0, 1).to(torch.uint8))

            gt_data_ = data.as_pandas_tensor_collection(object_labels=object_labels)
            gt_data_.infos["batch_im_id"] = batch_im_id
            gt_data.append(gt_data_)

            if hasattr(gt_data_, "poses_init"):
                initial_data_ = copy.deepcopy(gt_data_)
                initial_data_.poses = initial_data_.poses_init
                initial_data.append(initial_data_)

            gt_detections_ = copy.deepcopy(gt_data_)
            gt_detections_.infos["score"] = 1.0
            gt_detections.append(gt_detections_)

        return {
            "cameras": tc.PandasTensorCollection(
                infos=pd.DataFrame(cam_infos),
                K=torch.as_tensor(np.stack(K)),
            ),
            "rgb": torch.stack(rgb_images),
            "im_infos": im_infos,
            "gt_detections": tc.concatenate(gt_detections),
            "gt_data": tc.concatenate(gt_data),
            "initial_data": tc.concatenate(initial_data) if initial_data else None,
        }

    def as_pandas_tensor_collection(
        self,
        object_labels: Optional[List[str]] = None,
    ) -> SceneObservationTensorCollection:
        assert self.camera_data is not None
        assert self.object_datas is not None

        infos = []
        TWO = []
        bboxes = []
        masks = []
        TWC = torch.as_tensor(self.camera_data.TWC.matrix).float()
        TWC_init = None
        TWO_init = []

        if self.camera_data.TWC_init is not None:
            TWC_init = torch.as_tensor(self.camera_data.TWC_init.matrix).float()

        for obj_data in self.object_datas:
            if object_labels is not None and obj_data.label not in object_labels:
                continue
            infos.append(
                {
                    "label": obj_data.label,
                    "scene_id": self.infos.scene_id,
                    "view_id": self.infos.view_id,
                    "visib_fract": getattr(obj_data, "visib_fract", 1),
                }
            )
            TWO.append(torch.tensor(obj_data.TWO.matrix).float())
            bboxes.append(torch.tensor(obj_data.bbox_modal).float())

            if self.binary_masks is not None:
                masks.append(torch.tensor(self.binary_masks[obj_data.unique_id]).float())

            if obj_data.TWO_init:
                TWO_init.append(torch.tensor(obj_data.TWO_init.matrix).float())

        TWO = torch.stack(TWO)
        TCO = torch.linalg.inv(TWC).unsqueeze(0) @ TWO
        infos = pd.DataFrame(infos)
        data = tc.PandasTensorCollection(
            infos=infos,
            TCO=TCO,
            TWO=TWO,
            bboxes=torch.stack(bboxes),
            poses=TCO,
            K=torch.tensor(self.camera_data.K).unsqueeze(0).expand([len(infos), -1, -1]),
        )

        if masks:
            data.register_tensor("masks", torch.stack(masks))
        if TWO_init:
            TCO_init = torch.linalg.inv(TWC_init).unsqueeze(0) @ torch.stack(TWO_init)
            data.register_tensor("TCO_init", TCO_init)
            data.register_tensor("poses_init", TCO_init)
        return data
