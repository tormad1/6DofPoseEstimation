from __future__ import annotations

from typing import List

# Third Party
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# MegaPose
import src.megapose.utils.tensor_collection as tc
from src.megapose.datasets.scene_dataset import SceneObservation, ObjectData
from src.custom_megapose.web_scene_dataset import (
    WebSceneDataset,
)
from src.custom_megapose.web_scene_dataset import IterableWebSceneDataset
from src.utils.bbox import BoundingBox
from src.utils.logging import get_logger
from src.utils.inout import load_test_list_and_cnos_detections
from bop_toolkit_lib import pycoco_utils
from src.utils.dataset import LMO_ID_to_index

logger = get_logger(__name__)


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
        split = self.get_split_name(dataset_name)
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.transforms = transforms

        # load the dataset
        webdataset_dir = self.root_dir / self.dataset_name
        web_dataset = WebSceneDataset(webdataset_dir / split)
        self.web_dataloader = IterableWebSceneDataset(web_dataset, set_length=True)

        # depending on setting:
        # 1. localization: load target_objects
        # 2. detection: not target_objects
        assert test_setting in [
            "localization",
            "detection",
        ], f"{test_setting} not supported!"
        self.load_detections(test_setting=test_setting)

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

    def get_split_name(self, dataset_name):
        if dataset_name in ["hb", "tless"]:
            split = "test_primesense"
        else:
            split = "test"
        logger.info(f"Split: {split} for {dataset_name}!")
        return split

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
            if len(scene_obs.object_datas) == 0:
                gt_available = False
            else:
                gt_available = True
                gt_mapping = {
                    obj_data.label: obj_data for obj_data in scene_obs.object_datas
                }
            object_datas = []
            binary_masks = {}
            for idx, det in enumerate(dets):
                data = {}
                # use confidence score as visibility
                data["visib_fract"] = det["score"]
                data["bbox_modal"] = det["bbox"]
                data["bbox_amodal"] = det["bbox"]
                data["label"] = str(det["category_id"])
                data["unique_id"] = f"{idx+1}"

                if gt_available and data["label"] in gt_mapping:
                    obj_data = gt_mapping[data["label"]]
                    data["TWO"] = obj_data.TWO._T
                else:
                    data["TWO"] = [[1, 0, 0, 0], [0, 0, 0]]
                object_data = ObjectData.from_json(data)
                object_datas.append(object_data)

                # load mask
                binary_mask = pycoco_utils.rle_to_binary_mask(det["segmentation"])
                binary_masks[object_data.unique_id] = binary_mask
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
