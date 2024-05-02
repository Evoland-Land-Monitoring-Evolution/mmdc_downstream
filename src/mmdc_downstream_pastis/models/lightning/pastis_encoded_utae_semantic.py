#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for lai regression prediction """

import logging
from typing import Any

from mmdc_downstream_pastis.models.lightning.pastis_utae_semantic import (
    PastisUTAE,
    to_class_label,
)

from ..components.losses import compute_losses
from ..torch.utae import UTAE
from ..torch.utae_fusion import UTAEFusion

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logging.getLogger(__name__).setLevel(logging.INFO)


class MMDCPastisEncodedUTAE(PastisUTAE):
    """
    Pastis lightning module.
    Attributes:

    """

    def __init__(
        self,
        model: UTAE | UTAEFusion,
        metrics_list: list[str] = ["cross"],
        losses_list: list[str] = ["cross"],
        lr: float = 0.001,
        resume_from_checkpoint: str | None = None,
    ):
        super().__init__(model, metrics_list, losses_list, lr, resume_from_checkpoint)

    def step(self, batch: Any, stage: str = "train") -> Any:
        """
        One step.
        We compute logits and data classes for encoded PASTIS
        """
        x, data_masks, dates, gt, mask, patch_id = batch
        # batch_dict, data_masks_dict, doys_dict, target, mask, id_patch
        gt = gt.long()
        # out: OutUTAEForward = self.forward(x, batch_positions=dates)
        # logits = out.seg_map
        logits = self.forward(x, batch_positions=dates)
        logits_clip = logits[:, :, 32:-32, 32:-32].contiguous()
        gt_clip = gt[:, 32:-32, 32:-32].contiguous()
        losses = compute_losses(
            preds=logits_clip,
            target=gt_clip,
            mask=(gt_clip == -1),
            losses_list=self.losses_list,
        )
        self.iou_meter[stage].add(to_class_label(logits_clip), gt_clip)

        return losses
