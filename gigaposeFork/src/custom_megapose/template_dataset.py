from __future__ import annotations

# Standard Library
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from tqdm import tqdm

# MegaPose
from src.megapose.utils.tensor_collection import PandasTensorCollection
from src.custom_megapose.transform import Transform, ScaleTransform
from src.utils.pil import open_image
from src.utils.logging import get_logger

SceneObservationTensorCollection = PandasTensorCollection
Listbox = List[int]
ListPose = List[List[float]]
SingleDataJsonType = Union[str, float, ListPose, int, Listbox, Any]
DataJsonType = Union[Dict[str, SingleDataJsonType], List[SingleDataJsonType]]

logger = get_logger(__name__)


@dataclass
class TemplateData:
    label: str
    template_dir: str
    num_templates: int
    TWO_init: Transform
    pose_path: Optional[str] = None
    unique_id: Optional[int] = None
    TWO: Optional[List(Transform)] = None
    box_amodal: Optional[np.ndarray] = None  # (4, ) array [xmin, ymin, xmax, ymax]

    @staticmethod
    def from_dict(template_gt: DataJsonType) -> "TemplateData":
        assert isinstance(template_gt, dict)
        data = TemplateData(
            label=template_gt["label"],
            template_dir=str(template_gt["template_dir"]),
            pose_path=str(template_gt["pose_path"]),
            num_templates=int(template_gt["num_templates"]),
            TWO_init=ScaleTransform(scale_factor=template_gt["scale_factor"]),
        )
        return data

    def load_template(self, view_id, inplane=None):
        image_path = f"{self.template_dir}/{view_id:06d}.png"
        assert os.path.exists(image_path), f"{image_path} does not exist"
        rgba = open_image(image_path, inplane)
        box = rgba.getbbox()
        box_size = (box[2] - box[0], box[3] - box[1])
        if min(box_size) == 0:
            box = (0, 0, int(rgba.size[0]), int(rgba.size[1]))
            logger.warning(
                f"Template {image_path} has zero area, setting to null template"
            )
        data = {"rgba": np.array(rgba), "box": np.array(box)}
        return data

    def load_set_of_templates(
        self, view_ids, reload=False, inplanes=None, reset=True
    ):
        if inplanes is None:
            inplanes = [None for _ in view_ids]
        root_dir = os.path.dirname(self.template_dir)
        obj_id = os.path.basename(self.template_dir)

        preprocessed_file = f"{root_dir}/preprocessed/{int(obj_id):06d}.npz"
        if os.path.exists(preprocessed_file) and reload and not reset:
            data = np.load(preprocessed_file)
            rgba = torch.from_numpy(data["rgba"]).float()
            box = torch.from_numpy(data["box"]).long()
            return {"rgba": rgba, "box": box}
        else:
            os.makedirs(f"{root_dir}/preprocessed", exist_ok=True)
            data = {"rgba": [], "box": []}
            for view_id, inplane in zip(view_ids, inplanes):
                view_data = self.load_template(view_id, inplane=inplane)
                rgba = torch.from_numpy(view_data["rgba"] / 255).float()
                box = torch.from_numpy(view_data["box"]).long()
                data["rgba"].append(rgba)
                data["box"].append(box)
            data["rgba"] = torch.stack(data["rgba"]).permute(0, 3, 1, 2)
            data["box"] = torch.stack(data["box"])
            if reload:
                np.savez(
                    preprocessed_file,
                    rgba=data["rgba"].numpy(),
                    box=data["box"].numpy(),
                )
                logger.info(f"Preprocessed {preprocessed_file}")
        return data

    def apply_transform(self, transform, data):
        data["rgba"], data["M"] = transform(
            images=data["rgba"], boxes=data["box"], return_transform=True
        )
        return data

    def load_pose(self):
        poses = np.load(self.pose_path)
        poses = [Transform(poses[i]) * self.TWO_init for i in range(len(poses))]
        return torch.stack([pose.toTensor() for pose in poses])

    def read_test_mode(
        self,
    ):
        data = self.load_set_of_templates(view_ids=np.arange(0, self.num_templates))
        poses = self.load_pose()
        return data, poses


@dataclass
class TemplateDataset:
    def __init__(
        self,
        object_templates: List[TemplateData],
    ):
        self.list_object_templates = object_templates
        self.label_to_objects = {obj.label: obj for obj in object_templates}
        self.K = np.array(
            [572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]
        ).reshape((3, 3))
        if len(self.list_object_templates) != len(self.label_to_objects):
            raise RuntimeError("There are objects with duplicate labels")

    def __getitem__(self, idx: int) -> TemplateData:
        return self.list_object_templates[idx]

    def get_object_templates(self, label: str) -> TemplateData:
        return self.label_to_objects[label]

    def __len__(self) -> int:
        return len(self.list_object_templates)

    @property
    def objects(self) -> List[TemplateData]:
        """Returns a list of objects in this dataset."""
        return self.list_object_templates

    def filter_objects(self, selected_objects) -> "TemplateDataset":
        selected_object_ids = [
            selected_objects[unique_id]["label"] for unique_id in selected_objects
        ]
        list_objects = [
            obj
            for obj in self.list_object_templates
            if obj.label in selected_object_ids
        ]
        return TemplateDataset(list_objects)

    def from_config(
        model_infos: DataJsonType,
        config: DataJsonType,
    ) -> "TemplateDataset":
        template_datas = []
        for model_info in tqdm(model_infos):
            obj_id = model_info["obj_id"]
            template_metaData = {"label": str(obj_id)}
            template_metaData["num_templates"] = config.num_templates
            template_metaData["template_dir"] = os.path.join(
                config.dir, f"{obj_id:06d}"
            )
            pose_path = os.path.join(config.dir, config.pose_name)
            template_metaData["pose_path"] = pose_path.replace(
                "OBJECT_ID", f"{obj_id:06d}"
            )
            template_metaData["scale_factor"] = config.scale_factor
            template_data = TemplateData.from_dict(template_metaData)
            template_datas.append(template_data)
        logger.info(f"Loaded {len(template_datas)} template datas")
        return TemplateDataset(template_datas)
