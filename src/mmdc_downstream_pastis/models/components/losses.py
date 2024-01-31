""" Custom losses for downstream task """

import torch
from torch.nn import CrossEntropyLoss

from mmdc_singledate.models.components.losses import mask_and_flatten



def compute_losses(preds: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor,
                   losses_list: list[str] = ["cross"],
                   margin: int = 0) -> dict[str, torch.Tensor]:
    """Computes regression losses"""
    H, W = preds.shape[-2:]

    preds = mask_and_flatten(preds[:, margin:H - margin, margin:W - margin], mask)
    target = mask_and_flatten(target[:, margin:H - margin, margin:W - margin], mask)

    losses_dict = {}


    if "cross" in losses_list:
        cross = CrossEntropyLoss().to(preds.device)
        losses_dict["cross_entropy"] = cross(preds, target)


    return losses_dict
