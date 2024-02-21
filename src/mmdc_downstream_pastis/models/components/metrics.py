import torch
from mmdc_singledate.models.components.losses import mask_and_flatten
from torch.nn import CrossEntropyLoss


def compute_val_metrics(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    metrics_list: list[str],
) -> dict[str, torch.Tensor]:
    """Computes regression metrics"""

    if mask is not None:
        preds = mask_and_flatten(preds, mask)
        target = mask_and_flatten(target, mask)

    metrics_dict = {}
    if "cross_entropy" in metrics_list:
        cross = CrossEntropyLoss().to(preds.device)
        metrics_dict["cross_entropy"] = cross(preds, target)
    return metrics_dict
