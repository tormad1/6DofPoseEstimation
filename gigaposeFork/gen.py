from pathlib import Path
from PIL import Image
import json
import numpy as np
from pycocotools import mask as mask_utils

rgb = Path("gigaPose_datasets/datasets/T282/test_scenewise/000001/rgb")
out = Path(
    "gigaPose_datasets/datasets/default_detections/"
    "core19_model_based_unseen/cnos-fastsam/"
    "cnos-fastsam_T282-test_custom.json"
)

if not rgb.exists():
    raise FileNotFoundError(f"RGB folder not found: {rgb}")

out.parent.mkdir(parents=True, exist_ok=True)

detections = []
pngs = sorted(p for p in rgb.iterdir() if p.suffix.lower() == ".png")

if not pngs:
    raise RuntimeError(f"No PNG images found in: {rgb}")

for p in pngs:
    with Image.open(p) as im:
        w, h = im.size

    x = w // 4
    y = h // 4
    bw = w // 2
    bh = h // 2

    arr = np.zeros((h, w), dtype=np.uint8, order="F")
    arr[y:y + bh, x:x + bw] = 1

    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("ascii")

    detections.append(
        {
            "scene_id": 1,
            "image_id": int(p.stem),
            "category_id": 1,
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "score": 1.0,
            "segmentation": {
                "size": [int(h), int(w)],
                "counts": rle["counts"],
            },
            "time": 0.0,
        }
    )

out.write_text(json.dumps(detections, indent=2))
print(f"Wrote {out}")
print(f"Detections: {len(detections)}")
