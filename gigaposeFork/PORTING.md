# Porting notes

- Windows
- CPython embed
- CPU only
- RGB only
- top-1 pose
- ROI in, pose out

Use this import:

```python
from gigapose_runtime import create_runtime_from_paths
```

Build one runtime. Keep it alive.

```python
runtime = create_runtime_from_paths(
    checkpoint_path="./gigaPose_datasets/pretrained/gigaPose_v1.ckpt",
    dinov2_repo_dir="./dinov2",
    dataset_root_dir="./gigaPose_datasets/datasets",
    template_dir="./gigaPose_datasets/datasets/templates",
    dataset_name="T282",
    num_templates=162,
    cpu_threads=4,
    warmup=True,
)
```

Unity / native path:

- C++ DLL owns embedded CPython
- pass ROI bytes into Python
- pass full-frame camera intrinsics `K`
- pass ROI rect `bbox_xywh`
- get one pose dict back

Call:

```python
pose = runtime.run_roi_bytes(
    roi_bytes=roi_bytes,
    width=roi_width,
    height=roi_height,
    K=K,
    bbox_xywh=[x, y, w, h],
    object_id=1,
    channels=4,
    channel_order="BGRA",
    stride=row_stride_bytes,
)
```

Input contract:

- image dtype: `uint8`
- channel order: `RGB`, `BGR`, `RGBA`, `BGRA`
- no RGB-D
- no batch stream constructor per frame

Output contract:

- `None` if no pose
- else one dict with `R`, `t`, `score`, `object_id`
