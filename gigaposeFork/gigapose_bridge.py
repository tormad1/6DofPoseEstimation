from __future__ import annotations

import math
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
SITE_PACKAGES_DIR = REPO_DIR / ".venv" / "Lib" / "site-packages"
if SITE_PACKAGES_DIR.exists():
    site_packages = str(SITE_PACKAGES_DIR)
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

from gigapose_runtime import create_runtime_from_paths


_RUNTIME = None


def _rotation_matrix_to_quaternion(rotation_matrix):
    r00, r01, r02 = [float(value) for value in rotation_matrix[0]]
    r10, r11, r12 = [float(value) for value in rotation_matrix[1]]
    r20, r21, r22 = [float(value) for value in rotation_matrix[2]]

    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("rotation matrix produced zero-length quaternion")

    return [qx / norm, qy / norm, qz / norm, qw / norm]


def init_runtime(
    repo_root,
    dataset_name="T282",
    num_templates=162,
    cpu_threads=4,
    warmup=False,
):
    global _RUNTIME

    repo_root = Path(repo_root).resolve()
    gigapose_dir = repo_root / "gigaposeFork"
    if not gigapose_dir.exists():
        raise FileNotFoundError(f"gigaposeFork not found under {repo_root}")

    checkpoint_path = gigapose_dir / "gigaPose_datasets" / "pretrained" / "gigaPose_v1.ckpt"
    dinov2_repo_dir = gigapose_dir / "dinov2"
    dataset_root_dir = gigapose_dir / "gigaPose_datasets" / "datasets"
    template_dir = dataset_root_dir / "templates"

    _RUNTIME = create_runtime_from_paths(
        checkpoint_path=checkpoint_path,
        dinov2_repo_dir=dinov2_repo_dir,
        dataset_root_dir=dataset_root_dir,
        template_dir=template_dir,
        dataset_name=dataset_name,
        num_templates=num_templates,
        cpu_threads=cpu_threads,
        warmup=warmup,
    )
    return {
        "dataset_name": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "template_dir": str(template_dir),
    }


def run_roi_rgba(
    roi_bytes,
    width,
    height,
    stride,
    K,
    bbox_xywh,
    object_id=1,
    scene_id=1,
    im_id=1,
):
    if _RUNTIME is None:
        raise RuntimeError("runtime not initialized")

    pose = _RUNTIME.run_roi_bytes(
        roi_bytes=roi_bytes,
        width=width,
        height=height,
        K=K,
        bbox_xywh=bbox_xywh,
        object_id=object_id,
        scene_id=scene_id,
        im_id=im_id,
        channels=4,
        channel_order="RGBA",
        stride=stride,
    )
    if pose is None:
        return None

    return {
        "translation": [float(value) for value in pose["t"]],
        "rotation": _rotation_matrix_to_quaternion(pose["R"]),
        "score": float(pose["score"]),
        "object_id": int(pose["object_id"]),
    }
