""" Custom losses for downstream task """

import torch
from einops import rearrange
from torch.nn import CrossEntropyLoss


def compute_losses(
    preds: torch.Tensor,
    target: torch.Tensor,
    losses_list: list[str] = ["cross_entropy"],
) -> dict[str, torch.Tensor]:
    """Computes classification losses"""

    preds = rearrange(preds, "b c h w -> (b h w) c")
    target = rearrange(target, "b h w -> (b h w)")

    losses_dict = {}

    if "cross_entropy" in losses_list:
        cross = CrossEntropyLoss(reduction="mean", ignore_index=0).to(preds.device)
        losses_dict["cross_entropy"] = cross(preds, target)
    return losses_dict
