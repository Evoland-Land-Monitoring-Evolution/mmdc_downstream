""" Custom losses for downstream task """

import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from mmdc_singledate.models.components.losses import mask_and_flatten


def compute_losses(preds: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor,
                   losses_list: list[str],
                   margin: int = 0) -> dict[str, torch.Tensor]:
    """Computes regression losses"""
    H, W = preds.shape[-2:]

    mask = mask[:, :, margin:H - margin, margin:W - margin]

    preds = mask_and_flatten(preds[:, :, margin:H - margin, margin:W - margin], mask)
    target = mask_and_flatten(target[:, :, margin:H - margin, margin:W - margin], mask)

    losses_dict = {}

    if "MSE" in losses_list or "mse" in losses_list:
        relative_squared_error = MeanSquaredError().to(preds.device)
        losses_dict["MSE"] = relative_squared_error(preds, target)

    if "RMSE" in losses_list or "rmse" in losses_list:
        rmse = MeanSquaredError(squared=False).to(preds.device)
        losses_dict["RMSE"] = rmse(preds, target)

    if "L1" in losses_list or "MAE" in losses_list or "mae" in losses_list:
        mean_absolute_error = MeanAbsoluteError().to(preds.device)
        losses_dict["MAE"] = mean_absolute_error(preds, target)

    return losses_dict
