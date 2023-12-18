from typing import Any

import torch
from mmdc_singledate.datamodules.datatypes import MMDCBatch
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch
from mmdc_singledate.utils.train_utils import standardize_data

from mmdc_downstream.mmdc_model.model import PretrainedMMDC
from .base import MMDCDownstreamBaseLitModule
from ..components.losses import mmdc_mse
from ..datatypes import OutputLAI
from ..torch.lai_regression import MMDCDownstreamRegressionModule
from ...snap.components.compute_bio_var import predict_variable_from_tensors, stand_lai, unstand_lai, prepare_s2_image
from ...snap.lai_snap import BVNET


class MMDCDownstreamRegressionLitModule(MMDCDownstreamBaseLitModule):
    """
    LAI regression lightning module.
    Attributes:
        model: regression model
        model_snap: Fixed weights SNAP model for LAI GT computation
        model_mmdc: Pretrained MMDC model for latent representations
    """

    def __init__(
            self,
            model: MMDCDownstreamRegressionModule,
            model_snap: BVNET,
            model_mmdc: PretrainedMMDC | None = None,
            input_data: str = "experts",
            lr: float = 0.001,
            resume_from_checkpoint: str | None = None,
    ):
        super().__init__(model, model_mmdc, lr, resume_from_checkpoint)

        self.model_snap = model_snap
        self.model_snap.set_snap_weights()
        self.input_data = input_data

        self.margin = self.model_mmdc.model_mmdc.nb_cropped_hw if self.model_mmdc is not None else 0


    def get_regression_input(self, batch: MMDCBatch) -> torch.Tensor:
        """
        Prepare input for downstream model.
        It can be normalized S1/S2 data or one of latent representations
        produced by MMDC model.
        """
        # [experts, lat_S1, lat_S2, S2, S1_asc, S1_desc]
        if self.input_data == "experts":
            return self.model_mmdc.get_latent_mmdc(batch).latent_experts_mu
        elif self.input_data == "lat_S1":
            return self.model_mmdc.get_latent_mmdc(batch).latent_S1_mu
        elif self.input_data == "lat_S2":
            return self.model_mmdc.get_latent_mmdc(batch).latent_S2_mu
        elif self.input_data == "S2":
            s2_x = standardize_data(
                batch.s2_x,
                shift=self.stats.sen2.shift.type_as(
                    batch.s2_x),
                scale=self.stats.sen2.shift.type_as(
                    batch.s2_x),
            )
            return prepare_s2_image(s2_x, batch.s2_a, reshape=False)
        else:
            s1_x = standardize_data(
                batch.s1_x,
                shift=self.stats.sen1.shift.type_as(
                    batch.s1_x),
                scale=self.self.stats.sen1.shift.type_as(
                    batch.s1_x),
            )
            if self.input_data == "S1_asc":
                return s1_x[:, :3]
            else:  # "S1_desc":
                return s1_x[:, 3:]

    def step(self, batch: Any) -> Any:
        """One step"""
        batch: MMDCBatch = destructure_batch(batch)
        margin = self.model_mmdc.model_mmdc.nb_cropped_hw
        lai_gt = self.compute_gt(batch)
        reg_input = self.get_regression_input(batch)
        lai_pred = self.forward(reg_input)
        H, W = lai_pred.shape[-2:]
        loss_mse = mmdc_mse(lai_pred[:, :, self.margin:H-self.margin, self.margin:W-self.margin],
                            lai_gt[:, :, self.margin:H-self.margin, self.margin:W-self.margin],
                            batch.s2_m[:, :, self.margin:H-self.margin, self.margin:W-self.margin])
        return loss_mse

    def training_step(  # pylint: disable=arguments-differ
            self,
            batch: Any,
            batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Training step. Step and return loss."""
        torch.autograd.set_detect_anomaly(True)
        reg_loss = self.step(batch)
        self.log(
            "train/loss",
            reg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": reg_loss}

    def validation_step(  # pylint: disable=arguments-differ
            self,
            batch: Any,
            batch_idx: int,  # pylint: disable=unused-argument
            prefix: str = "val",
    ) -> dict[str, Any]:
        """Validation step. Step and return loss."""
        reg_loss = self.step(batch)

        self.log(
            f"{prefix}/loss",
            reg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": reg_loss}

    def test_step(  # pylint: disable=arguments-differ
            self,
            batch: Any,
            batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Test step. Step and return loss. Delegate to validation step"""
        return self.validation_step(batch, batch_idx, prefix="test")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        return self.model.forward(data)

    def predict(self, batch: MMDCBatch) -> OutputLAI:
        self.model.eval()
        lai_gt = self.compute_gt(batch)
        reg_input = self.get_regression_input(batch)
        lai_pred = unstand_lai(self.forward(reg_input))
        return OutputLAI(lai_pred, reg_input, unstand_lai(lai_gt))

    def compute_gt(self, batch: MMDCBatch, stand: bool = True) -> torch.Tensor:
        """Compute LAI GT data wiht standardisation or not"""
        gt = predict_variable_from_tensors(batch.s2_x, batch.s2_a, batch.s2_m,
                                           self.model_snap)
        if stand:
            return stand_lai(gt)
        return gt

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.learning_rate)

        training_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4, T_mult=2, eta_min=0, last_epoch=-1)
        # scheduler = {
        #     "scheduler": training_scheduler,
        #     "interval": "epoch",
        #     "monitor": "val/mse_loss",
        #     "frequency": 1,
        # }
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
        }
