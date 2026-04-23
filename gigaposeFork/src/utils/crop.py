import torch
from src.utils.logging import get_logger
from src.utils.bbox import BoundingBox
import torch.nn.functional as F

logger = get_logger(__name__)


class CropResizePad:
    def __init__(self, target_size=224, patch_size=14):
        self.target_size = target_size
        self.patch_size = patch_size

    def __call__(self, xyxy_boxes, images):
        batch_size = xyxy_boxes.shape[0]
        device = xyxy_boxes.device
        bbox_sizes = BoundingBox(xyxy_boxes).get_box_size()
        scales = self.target_size / torch.max(bbox_sizes, dim=-1)[0]

        out_data = {"M": [], "images": []}
        for i in range(batch_size):
            image = images[i]
            bbox, bbox_size, scale = xyxy_boxes[i], bbox_sizes[i], scales[i]

            M_crop = torch.eye(3, device=device)
            M_resize_pad = torch.eye(3, device=device)

            # crop and scale
            image = image[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
            M_crop[:2, 2] = -bbox[:2]

            image = F.interpolate(image.unsqueeze(0), scale_factor=scale.item())[0]
            M_resize_pad[:2, :2] *= scale

            if image.shape[-1] / image.shape[-2] != 1:
                pad_top = (self.target_size - image.shape[-2]) // 2
                pad_bottom = self.target_size - image.shape[-2] - pad_top
                pad_bottom = max(pad_bottom, 0)

                pad_left = (self.target_size - image.shape[-1]) // 2
                pad_left = max(pad_left, 0)
                pad_right = self.target_size - image.shape[-1] - pad_left

                image = F.pad(image, [pad_left, pad_right, pad_top, pad_bottom])
                M_resize_pad[:2, 2] = torch.tensor([pad_left, pad_top])

            M = torch.matmul(M_resize_pad, M_crop)

            # sometimes, 1 pixel is missing due to rounding, so interpolate again
            image = F.interpolate(
                image.unsqueeze(0), size=(self.target_size, self.target_size)
            )[0]

            out_data["M"].append(M)
            out_data["images"].append(image)

        out_data["M"] = torch.stack(out_data["M"])
        out_data["images"] = torch.stack(out_data["images"])
        return out_data
