""" Custom losses for downstream task """

import torch
from einops import rearrange
from torch.nn import CrossEntropyLoss


def compute_losses(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    losses_list: list[str] = ["cross_entropy"],
) -> dict[str, torch.Tensor]:
    """Computes classification losses"""

    if mask is not None:
        batch, classes, _, _ = preds.shape

        preds = rearrange(preds, "b c h w -> (b h w) c")
        mask = rearrange(mask, "b h w -> (b h w)")
        target = rearrange(target, "b h w -> (b h w)")
        preds = preds[~mask]
        target = target[~mask]

    losses_dict = {}

    if "cross_entropy" in losses_list:
        cross = CrossEntropyLoss().to(preds.device)
        losses_dict["cross_entropy"] = cross(preds, target)

    return losses_dict
