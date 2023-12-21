import torch
from mmdc_singledate.models.components.losses import mask_and_flatten
from torchmetrics.regression import RelativeSquaredError, R2Score, MeanAbsoluteError, MeanAbsolutePercentageError


def compute_val_metrics(preds: torch.Tensor,
                        target: torch.Tensor,
                        mask: torch.Tensor,
                        margin: int) -> dict[str, torch.Tensor]:
    """Computes regression metrics"""
    H, W = preds.shape[-2:]

    mask = mask[:, :, margin:H - margin, margin:W - margin]

    preds = mask_and_flatten(preds[:, :, margin:H - margin, margin:W - margin], mask)
    target = mask_and_flatten(target[:, :, margin:H - margin, margin:W - margin], mask)

    metrics_dict = {}
    relative_squared_error = RelativeSquaredError().to(preds.device)
    metrics_dict["RSE"] = relative_squared_error(preds, target)

    r2score = R2Score().to(preds.device)
    metrics_dict["R2"] = r2score(preds, target)

    mean_absolute_error = MeanAbsoluteError().to(preds.device)
    metrics_dict["MAE"] = mean_absolute_error(preds, target)

    mean_abs_percentage_error = MeanAbsolutePercentageError().to(preds.device)
    metrics_dict["MAPE"] = mean_abs_percentage_error(preds, target)

    return metrics_dict
