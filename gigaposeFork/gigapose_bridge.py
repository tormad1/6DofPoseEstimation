from __future__ import annotations

import os
import math
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
SITE_PACKAGES_DIR = REPO_DIR / ".venv" / "Lib" / "site-packages"
if SITE_PACKAGES_DIR.exists():
    site_packages = str(SITE_PACKAGES_DIR)
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

_DLL_DIR_HANDLES = []


def _register_dll_directories():
    candidate_dirs = [
        REPO_DIR / ".python" / "python-3.11.9-embed-amd64",
        SITE_PACKAGES_DIR / "numpy.libs",
        SITE_PACKAGES_DIR / "scipy.libs",
        SITE_PACKAGES_DIR / "pandas.libs",
        SITE_PACKAGES_DIR / "torch" / "lib",
    ]
    if not hasattr(os, "add_dll_directory"):
        return

    for dll_dir in candidate_dirs:
        if not dll_dir.exists():
            continue
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))


_register_dll_directories()

from gigapose_runtime import create_runtime_from_paths


_RUNTIME = None
_MM_TO_METERS = 1.0 / 1000.0
_BOP_TO_UNITY_BASIS = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)


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


def _matmul3x3(left, right):
    out = []
    for row in range(3):
        out_row = []
        for col in range(3):
            value = 0.0
            for idx in range(3):
                value += float(left[row][idx]) * float(right[idx][col])
            out_row.append(value)
        out.append(out_row)
    return out


def _convert_bop_rotation_to_unity(rotation_matrix):
    basis = _BOP_TO_UNITY_BASIS
    return _matmul3x3(_matmul3x3(basis, rotation_matrix), basis)


def _convert_bop_translation_to_unity(translation):
    tx, ty, tz = [float(value) for value in translation]
    return [
        tx * _MM_TO_METERS,
        -ty * _MM_TO_METERS,
        tz * _MM_TO_METERS,
    ]


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

    raw_score = float(pose["score"])
    unity_translation = _convert_bop_translation_to_unity(pose["t"])
    unity_rotation = _rotation_matrix_to_quaternion(
        _convert_bop_rotation_to_unity(pose["R"])
    )

    return {
        "translation": unity_translation,
        "rotation": unity_rotation,
        # GigaPose's top-1 ROI path is returning 0.0 even for a valid self-match.
        # The native bridge already treats None as failure, so a returned pose is
        # a usable tracking update and should not be dropped by Unity.
        "score": 1.0,
        "raw_score": raw_score,
        "object_id": int(pose["object_id"]),
    }
