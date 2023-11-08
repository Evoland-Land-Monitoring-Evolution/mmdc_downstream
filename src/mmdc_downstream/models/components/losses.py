""" Custom losses for MMDC """

import torch
import torch.nn.functional as F


def mmdc_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Wrapper around Pytorch's MSE which can optionally take
    a validity mask

    Args:
       pred: torch.Tensor the tensor containing the predictions
       target: torch:Tensor the tensor containing the targets
       mask: torch.Tensor | None the validity mask of pixels to be taken into account
             for the computation

    Returns:
       torch.Tensor: the masked MSE loss
    """

    validity_ratio = 1.0
    if mask is not None:
        pred = mask_and_flatten(pred, mask)
        target = mask_and_flatten(target, mask)
    return F.mse_loss(pred, target) * validity_ratio


def mask_and_flatten(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask the data tensor and flatten the spatial dimensions"""
    if mask.shape[1] == 2:
        return mask_and_flatten_s1(data, mask)
    return mask_and_flatten_s2(data, mask)


def mask_and_flatten_s1(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask the S1 data tensor and flatten the spatial dimensions"""
    s1_bands = data.shape[1] // 2
    asc_data = data[:, :s1_bands, :, :]
    asc_mask = mask[:, :1, :, :].repeat(1, s1_bands, 1, 1) < 1
    asc_data = asc_data.transpose(0, 1).transpose(1, -1).flatten(1, -1)
    asc_mask = asc_mask.transpose(0, 1).transpose(1, -1).flatten(1, -1)
    assert asc_data.shape == asc_mask.shape
    desc_data = data[:, s1_bands:, :, :]
    desc_mask = mask[:, 1:, :, :].repeat(1, s1_bands, 1, 1) < 1
    desc_data = desc_data.transpose(0, 1).transpose(1, -1).flatten(1, -1)
    desc_mask = desc_mask.transpose(0, 1).transpose(1, -1).flatten(1, -1)
    assert desc_data.shape == desc_mask.shape
    asc_data = (
        asc_data.flatten()[asc_mask.flatten()].reshape(s1_bands, -1).transpose(0, 1)
    )
    desc_data = (
        desc_data.flatten()[desc_mask.flatten()].reshape(s1_bands, -1).transpose(0, 1)
    )

    return torch.cat([asc_data, desc_data], dim=0)


def mask_and_flatten_s2(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask the S1 data tensor and flatten the spatial dimensions"""
    new_mask = mask.expand_as(data) < 1
    masked_data = data.flatten()[new_mask.flatten()].reshape(-1, data.shape[1])
    return masked_data


def mask_data(data: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Mask the data tensor and and return de validity ratio"""
    if mask.shape[1] == 2:
        return mask_s1_like(data, mask)
    return mask_s2_like(data, mask)


def mask_s1_like(data: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Mask data with S1 shape for loss computation"""
    s1_bands = data.shape[1] // 2
    s1_x = data.clone()
    s1_x[:, :s1_bands, ...] = s1_x[:, :s1_bands, ...] * (1 - mask[:, 0, ...]).unsqueeze(
        1
    )
    s1_x[:, s1_bands:, ...] = s1_x[:, s1_bands:, ...] * (1 - mask[:, 1, ...]).unsqueeze(
        1
    )
    valid_ratio = 1.0 - mask.sum() / (mask.shape[2] * mask.shape[3] * 2)
    return s1_x, valid_ratio


def mask_s2_like(data: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Mask data with S2 shape for loss computation"""
    s2_x = data.clone()
    s2_x = s2_x * (1 - mask)
    valid_ratio = 1.0 - mask.sum() / (mask.shape[2] * mask.shape[3])
    return s2_x, valid_ratio
