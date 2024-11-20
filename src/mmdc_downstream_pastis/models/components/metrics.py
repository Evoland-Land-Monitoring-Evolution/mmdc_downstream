import torch
from einops import rearrange
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassF1Score


def compute_val_metrics(
    preds: torch.Tensor,
    target: torch.Tensor,
    metrics_list: list[str],
) -> dict[str, torch.Tensor]:
    """Computes regression metrics"""

    preds = rearrange(preds, "b c h w -> (b h w) c")
    target = rearrange(target, "b h w -> (b h w)")

    metrics_dict = {}

    if "cross_entropy" in metrics_list:
        cross = CrossEntropyLoss(reduction="mean", ignore_index=0).to(preds.device)
        metrics_dict["cross_entropy"] = cross(preds, target)

    if "f1" in metrics_list:
        f1_score = MulticlassF1Score(ignore_index=0, num_classes=19, average=None).to(
            preds.device
        )

        f1 = f1_score(preds, target)

        for i, score in enumerate(f1):
            metrics_dict[f"f1_{i}"] = score

    return metrics_dict
