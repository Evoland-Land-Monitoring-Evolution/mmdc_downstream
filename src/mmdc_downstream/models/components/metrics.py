import torch
from mmdc_singledate.models.components.losses import mask_and_flatten
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    R2Score,
    RelativeSquaredError,
)


def compute_val_metrics(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metrics_list: list[str],
    margin: int = 0,
) -> dict[str, torch.Tensor]:
    """Computes regression metrics"""
    H, W = preds.shape[-2:]

    mask = mask[:, :, margin : H - margin, margin : W - margin]

    preds = mask_and_flatten(
        preds[:, :, margin : H - margin, margin : W - margin], mask
    )
    target = mask_and_flatten(
        target[:, :, margin : H - margin, margin : W - margin], mask
    )

    metrics_dict = {}
    if "RSE" in metrics_list or "rse" in metrics_list:
        relative_squared_error = RelativeSquaredError().to(preds.device)
        metrics_dict["RSE"] = relative_squared_error(preds, target)

    if "R2" in metrics_list or "r2" in metrics_list:
        r2score = R2Score().to(preds.device)
        metrics_dict["R2"] = r2score(preds, target)

    if "MAE" in metrics_list or "mae" in metrics_list:
        mean_absolute_error = MeanAbsoluteError().to(preds.device)
        metrics_dict["MAE"] = mean_absolute_error(preds, target)

    if "MAPE" in metrics_list or "mape" in metrics_list:
        mean_abs_percentage_error = MeanAbsolutePercentageError().to(preds.device)
        metrics_dict["MAPE"] = mean_abs_percentage_error(preds, target)

    return metrics_dict
