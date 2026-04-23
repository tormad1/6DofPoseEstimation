import torch
from einops import repeat


def affine_torch(rotation, scale=None, translation=None):
    if len(rotation.shape) == 2:
        """
        Create 2D affine transformation matrix
        """
        M = torch.eye(3, device=scale.device, dtype=scale.dtype)
        M[:2, :2] = rotation
        if scale is not None:
            M[:2, :2] *= scale
        if translation is not None:
            M[:2, 2] = translation
        return M
    else:
        Ms = torch.eye(3, device=scale.device, dtype=scale.dtype)
        Ms = Ms.unsqueeze(0).repeat(rotation.shape[0], 1, 1)
        Ms[:, :2, :2] = rotation
        if scale is not None:
            Ms[:, :2, :2] *= scale.unsqueeze(1).unsqueeze(1)
        if translation is not None:
            Ms[:, :2, 2] = translation
        return Ms


def homogenuous(pixel_points):
    """
    Convert pixel coordinates to homogenuous coordinates
    """
    device = pixel_points.device
    if len(pixel_points.shape) == 2:
        one_vector = torch.ones(pixel_points.shape[0], 1).to(device)
        return torch.cat([pixel_points, one_vector], dim=1)
    elif len(pixel_points.shape) == 3:
        one_vector = torch.ones(pixel_points.shape[0], pixel_points.shape[1], 1).to(
            device
        )
        return torch.cat([pixel_points, one_vector], dim=2)
    else:
        raise NotImplementedError


def inverse_affine(M):
    """
    Inverse 2D affine transformation matrix of cropping
    """
    if len(M.shape) == 2:
        M = M.unsqueeze(0)
    if len(M.shape) == 3:
        assert (M[:, 1, 0] == 0).all() and (M[:, 0, 1] == 0).all()
        assert (M[:, 0, 0] == M[:, 1, 1]).all(), f"M: {M}"

        scale = M[:, 0, 0]
        M_inv = torch.eye(3, device=M.device, dtype=M.dtype)
        M_inv = M_inv.unsqueeze(0).repeat(M.shape[0], 1, 1)
        M_inv[:, 0, 0] = 1 / scale  # scale
        M_inv[:, 1, 1] = 1 / scale  # scale
        M_inv[:, :2, 2] = -M[:, :2, 2] / scale.unsqueeze(1)  # translation
    else:
        raise ValueError("M must be 2D or 3D")
    return M_inv


def apply_affine(M, points):
    """
    M: (N, 3, 3)
    points: (N, 2)
    """
    if len(points.shape) == 2:
        transformed_points = torch.einsum(
            "bhc,bc->bh",
            M,
            homogenuous(points),
        )  # (N, 3)
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    elif len(points.shape) == 3:
        transformed_points = torch.einsum(
            "bhc,bnc->bnh",
            M,
            homogenuous(points),
        )
        transformed_points = transformed_points[:, :, :2] / transformed_points[:, :, 2:]
    else:
        raise NotImplementedError
    return transformed_points


def normalize_affine_transform(transforms):
    """
    Input: Affine transformation
    Output: Normalized affine transformation
    """
    norm_transforms = torch.zeros_like(transforms)
    norm_transforms[:, :, 2, 2] = 1

    scale = torch.norm(transforms[:, :, :2, 0], dim=2)
    scale = repeat(scale, "b n -> b n h w", h=2, w=2)

    norm_transforms[:, :, :2, :2] = transforms[:, :, :2, :2] / scale
    return norm_transforms
