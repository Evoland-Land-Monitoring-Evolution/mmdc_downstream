""" Custom losses for downstream task """

import torch
from mmdc_singledate.models.components.losses import mask_and_flatten
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from ...snap.lai_snap import denormalize


def weighted_mse_loss(pred, target, weight):
    """weighted mse loss"""
    return (weight * (pred - target) ** 2).sum() / weight.sum()


def weighted_rmse_loss(pred, target, weight):
    """weighted rmse loss"""
    return weighted_mse_loss(pred, target, weight).sqrt()


def weighted_mae_loss(pred, target, weight):
    """weighted mae loss"""
    return (weight * torch.abs(pred - target)).sum() / weight.sum()


def compute_losses(preds: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor,
                   losses_list: list[str],
                   margin: int = 0,
                   bin_weights: torch.Tensor | None = None,
                   denorm_min_max: tuple[torch.Tensor, torch.Tensor] = None,
                   ) -> dict[str, torch.Tensor]:
    """Computes regression losses"""
    H, W = preds.shape[-2:]     # pylint: : disable=C0103

    mask = mask[:, :, margin:H - margin, margin:W - margin]

    preds = mask_and_flatten(preds[:, :, margin:H - margin, margin:W - margin], mask)
    target = mask_and_flatten(target[:, :, margin:H - margin, margin:W - margin], mask)

    losses_dict = {}

    if bin_weights is not None:
        target_denorm = denormalize(target, denorm_min_max[0], denorm_min_max[1])
        target_bins = torch.floor(target_denorm * 10).int()
        del target_denorm
        target_bins[target_bins > 149] = 149
        weights = bin_weights[target_bins]
        del target_bins

        if "MSE" in losses_list or "mse" in losses_list:
            losses_dict["MSE"] = weighted_mse_loss(preds, target, weights)

        if "RMSE" in losses_list or "rmse" in losses_list:
            losses_dict["RMSE"] = weighted_rmse_loss(preds, target, weights)

        if "L1" in losses_list or "MAE" in losses_list or "mae" in losses_list:
            losses_dict["MAE"] = weighted_mae_loss(preds, target, weights)

        return losses_dict


    if "MSE" in losses_list or "mse" in losses_list:
        mse = MeanSquaredError().to(preds.device)
        losses_dict["MSE"] = mse(preds, target)  # pylint: disable=E1102

    if "RMSE" in losses_list or "rmse" in losses_list:
        rmse = MeanSquaredError(squared=False).to(preds.device)
        losses_dict["RMSE"] = rmse(preds, target)   # pylint: disable=E1102

    if "L1" in losses_list or "MAE" in losses_list or "mae" in losses_list:
        mae = MeanAbsoluteError().to(preds.device)
        losses_dict["MAE"] = mae(preds, target)     # pylint: disable=E1102

    if "Huber" in losses_list:
        huber = torch.nn.HuberLoss().to(preds.device)
        losses_dict["Huber"] = huber(preds, target)     # pylint: disable=E1102

    return losses_dict
