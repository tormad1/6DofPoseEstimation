"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import numpy as np
import torch
from scipy.spatial.transform import Rotation


class Transform:
    """Small SE(3) helper for the inference path."""

    def __init__(self, *args):
        """
        - Transform(T): a (4, 4) homogeneous matrix.
        - Transform(rotation, translation): rotation is a xyzw quaternion or 3x3 matrix.
        """
        if len(args) == 1:
            T = _to_numpy(args[0])
            assert T.shape == (4, 4)
            self._matrix = T.astype(np.float64, copy=True)
            return

        if len(args) == 2:
            rotation, translation = args
            rotation_np = _to_numpy(rotation)

            if rotation_np.size == 4:
                R = Rotation.from_quat(rotation_np.reshape(4)).as_matrix()
            elif rotation_np.size == 9:
                R = rotation_np.reshape(3, 3)
            else:
                raise ValueError

            t = _to_numpy(translation).reshape(3)
            self._matrix = np.eye(4)
            self._matrix[:3, :3] = R
            self._matrix[:3, 3] = t
            return

        raise ValueError

    def __mul__(self, other: "Transform") -> "Transform":
        return Transform(self.matrix @ other.matrix)

    def __str__(self) -> str:
        return str(self._matrix)

    def toTensor(self) -> np.ndarray:
        return torch.from_numpy(self.matrix).float()

    @property
    def matrix(self) -> np.ndarray:
        """Returns 4x4 homogeneous matrix representations"""
        return self._matrix.copy()


class ScaleTransform(Transform):
    def __init__(self, scale_factor: float):
        scale_transform = np.eye(4)
        scale_transform[0, 0] *= scale_factor
        scale_transform[1, 1] *= scale_factor
        scale_transform[2, 2] *= scale_factor
        self._matrix = scale_transform


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
