#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks """
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch
from pytorch_lightning.callbacks import Callback
from sensorsio.utils import rgb_render
from torchmetrics import MeanSquaredError

from mmdc_downstream_lai.snap.lai_snap import denormalize

from ..models.lightning.lai_regression_conv import MMDCDLAILitModuleConv
from ..models.lightning.lai_regression_csv import MMDCLAILitModule

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class SampleInfo:
    """Information about the sample to be displayed"""

    batch_idx: int
    batch_size: int
    patch_margin: int
    current_epoch: int

    # class MMDCLAIScatterplotCallbackEpoch(Callback):
    #     """
    #     Callback to inspect the LAI image predicted from Latent Experts
    #     """
    #
    #     def __init__(
    #             self,
    #             save_dir: str,
    #     ):
    #         self.save_dir = save_dir
    #         self.loss_fn = MeanSquaredError(squared=False)  # root mean square error
    #
    #
    #
    #     def prepare_img_data(
    #             self,
    #             preds: torch.Tensor,
    #             gt: torch.Tensor,
    #             pos: list[torch.Tensor, torch.Tensor],
    #             reg_input: torch.Tensor,
    #             lai_min: torch.Tensor,
    #             lai_max: torch.Tensor,
    #     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #         margin = 28
    #         img, pix = pos
    #         img = img.to(int)
    #         pix = pix.to(int)
    #
    #         pred_img = torch.zeros((int(max(img)+1), 256 * 256))
    #         gt_img = torch.zeros((int(max(img)+1), 256 * 256))
    #         input_img = torch.zeros((int(max(img)+1), 256 * 256, reg_input.shape[1]))
    #
    #         pred_img[img, pix] = preds.cpu().detach()
    #         gt_img[img, pix] = gt.cpu().detach()
    #         input_img[img, pix] = reg_input.cpu().detach()
    #
    #         existing_img = torch.unique(img).cpu().detach()
    #
    #         pred_img = pred_img[existing_img]
    #         gt_img = gt_img[existing_img]
    #         input_img = input_img[existing_img]
    #
    #
    #
    #         pred_img = denormalize(pred_img.reshape(-1, 256, 256)[
    #                                :, margin:-margin, margin:-margin
    #                                ], lai_min, lai_max).numpy()
    #         gt_img = denormalize(gt_img.reshape(-1, 256, 256)[
    #                              :, margin:-margin, margin:-margin
    #                              ], lai_min, lai_max).numpy()
    #         input_img = input_img.reshape(-1, 256, 256,
    #                                       reg_input.shape[1]
    #                                       ).numpy().transpose(0, 3, 1, 2)[
    #                     :, :, margin:-margin, margin:-margin]
    #
    #         return pred_img, gt_img, input_img, existing_img.numpy()

    def on_validation_epoch_end(
        self,
        trainer: pl.trainer.Trainer,
        pl_module: MMDCLAILitModule,
    ) -> None:
        """Plot scatterplot"""

        self.lai_gt_pred_scatterplots(
            torch.Tensor(pl_module.pred_val),
            torch.Tensor(pl_module.gt_val),
            trainer.current_epoch,
            pl_module.model_snap.variable_min.cpu(),
            pl_module.model_snap.variable_max.cpu(),
        )


class MMDCLAIScatterplotCallbackStep(Callback):
    def __init__(
        self,
        save_dir: str,
    ):
        self.save_dir = save_dir
        self.loss_fn = MeanSquaredError(squared=False)  # root mean square error

    def prepare_img_data(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        pos: list[torch.Tensor, torch.Tensor],
        reg_input: torch.Tensor,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
        margin: int = 28,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img, pix = pos
        img = img.to(int)
        pix = pix.to(int)

        pred_img = torch.zeros((int(max(img) + 1), 256 * 256))
        gt_img = torch.zeros((int(max(img) + 1), 256 * 256))
        input_img = torch.zeros((int(max(img) + 1), 256 * 256, reg_input.shape[1]))

        pred_img[img, pix] = preds.cpu().detach()
        gt_img[img, pix] = gt.cpu().detach()
        input_img[img, pix] = reg_input.cpu().detach()

        existing_img = torch.unique(img).cpu().detach()

        pred_img = pred_img[existing_img]
        gt_img = gt_img[existing_img]
        input_img = input_img[existing_img]

        pred_img = denormalize(
            pred_img.reshape(-1, 256, 256)[:, margin:-margin, margin:-margin],
            lai_min,
            lai_max,
        ).numpy()
        gt_img = denormalize(
            gt_img.reshape(-1, 256, 256)[:, margin:-margin, margin:-margin],
            lai_min,
            lai_max,
        ).numpy()
        # input_img = input_img.reshape(-1, reg_input.shape[1], 256, 256)[1:-1].numpy()
        input_img = (
            input_img.reshape(-1, 256, 256, reg_input.shape[1])
            .numpy()
            .transpose(0, 3, 1, 2)[:, :, margin:-margin, margin:-margin]
        )

        return pred_img, gt_img, input_img, existing_img.numpy()

    def prepare_data(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        # idx = np.random.choice(len(preds), int(len(preds) * 0.05), replace=False)
        preds = denormalize(preds, lai_min, lai_max).cpu().detach()
        gt = denormalize(gt, lai_min, lai_max).cpu()
        rmse = np.round(self.loss_fn(preds, gt).item(), 2)

        logging.info("pred max " + str(preds.max()))
        logging.info("pred min " + str(preds.min()))

        return preds.numpy(), gt.numpy(), rmse

    def lai_gt_show_img(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        pos: list[torch.Tensor, torch.Tensor],
        reg_input: torch.Tensor,
        epoch: int,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
        batch_idx: int | None = None,
        train: bool = False,
        margin: int = 28,
    ) -> None:
        """Save the PNG image of the scatterplots of prediction vs gt"""

        pred_img, gt_img, input_img, img = self.prepare_img_data(
            preds, gt, pos, reg_input, lai_min, lai_max, margin
        )

        input_img = input_img[:, :3]

        plt.close()

        fig, axes = plt.subplots(
            nrows=len(pred_img),
            ncols=4,
            sharex=True,
            sharey=True,
            figsize=(13, len(pred_img) * 2.5),
        )
        fig.suptitle("Prediction Inspection", fontsize=20)

        for samp_idx in range(len(pred_img)):  # We iterate through samples to plot
            # input_denorm = (input_img[samp_idx].transpose(1, 2, 0) + 1)/2
            # axes[samp_idx, 0].imshow(input_denorm)

            axes[samp_idx, 0].imshow(
                rgb_render(input_img[samp_idx])[0], interpolation="bicubic"
            )
            zeros = (input_img[samp_idx][0] == 0).sum()
            axes[samp_idx, 0].set_title(f"Latent {img[samp_idx]} zeros={zeros}")

            min, mean, max = np.quantile(
                pred_img[samp_idx][pred_img[samp_idx] != 0], q=[0.01, 0.5, 0.99]
            )
            min_gt, mean_gt, max_gt = np.quantile(
                gt_img[samp_idx][gt_img[samp_idx] != 0], q=[0.01, 0.5, 0.99]
            )

            axes[samp_idx, 1].imshow(
                pred_img[samp_idx],
                cmap="RdYlGn",
                vmin=min_gt,
                vmax=max_gt,
                interpolation="bicubic",
            )
            axes[samp_idx, 1].set_title(f"Pred {int(min)} {int(max)}")

            axes[samp_idx, 2].imshow(
                gt_img[samp_idx],
                cmap="RdYlGn",
                vmin=min_gt,
                vmax=max_gt,
                interpolation="bicubic",
            )
            axes[samp_idx, 2].set_title(f"GT {int(min_gt)} {int(max_gt)}")

            error = np.abs(gt_img[samp_idx] - pred_img[samp_idx])
            axes[samp_idx, 3].imshow(error, cmap="Reds", interpolation="bicubic")
            axes[samp_idx, 3].set_title(f"Error, max={np.max(np.round(error, 1))}")

        if batch_idx is not None:
            if train:
                plt.savefig(self.save_dir + f"/train_mlp_img_{epoch}_{batch_idx}.png")
            else:
                plt.savefig(self.save_dir + f"/val_mlp_img_{epoch}_{batch_idx}.png")
        else:
            if train:
                plt.savefig(self.save_dir + f"/train_mlp_img_{epoch}.png")
            else:
                plt.savefig(self.save_dir + f"/val_mlp_img_{epoch}.png")

    def lai_gt_pred_scatterplots(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        epoch: int,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
        batch_idx: int | None = None,
    ) -> None:
        """Save the PNG image of the scatterplots of prediction vs gt"""
        preds, gt, rmse = self.prepare_data(
            preds,
            gt,
            lai_min,
            lai_max,
        )

        plt.close()
        plt.scatter(preds, gt, s=0.1)
        plt.plot(gt, gt)
        plt.title("loss=" + str(rmse))
        plt.xlabel("pred")
        plt.ylabel("gt")

        if batch_idx is not None:
            plt.savefig(self.save_dir + f"/val_mlp_{epoch}_{batch_idx}.png")
        else:
            plt.savefig(self.save_dir + f"/val_mlp_{epoch}.png")

    def on_validation_batch_end(
        self,
        trainer: pl.trainer.Trainer,
        pl_module: MMDCLAILitModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx < 3:
            reg_input, lai_gt, img, pix = batch
            lai_pred = pl_module.model.forward(reg_input)
            logger.info(f"{reg_input.shape=}")
            logger.info(f"{lai_gt.shape=}")
            logger.info(f"{lai_pred.shape=}")
            self.lai_gt_show_img(
                lai_pred.reshape(-1),
                lai_gt.reshape(-1),
                [img, pix],
                reg_input,
                trainer.current_epoch,
                pl_module.model_snap.variable_min.cpu(),
                pl_module.model_snap.variable_max.cpu(),
                batch_idx,
            )

            self.lai_gt_pred_scatterplots(
                lai_pred.reshape(-1),
                lai_gt.reshape(-1),
                trainer.current_epoch,
                pl_module.model_snap.variable_min.cpu(),
                pl_module.model_snap.variable_max.cpu(),
                batch_idx,
            )

        # def on_train_batch_end(
        #         self,
        #         trainer: pl.trainer.Trainer,
        #         pl_module: MMDCLAILitModule,
        #         outputs: Any,
        #         batch: torch.Tensor,
        #         batch_idx: int,
        #         dataloader_idx: int = 0,
        # ) -> None:
        #     reg_input, lai_gt, img, pix = batch
        #     lai_pred = pl_module.model.forward(reg_input)
        #     self.lai_gt_show_img(
        #         lai_pred.reshape(-1),
        #         lai_gt.reshape(-1),
        #         [img, pix],
        #         reg_input,
        #         trainer.current_epoch,
        #         pl_module.model_snap.variable_min.cpu(),
        #         pl_module.model_snap.variable_max.cpu(),
        #         batch_idx,
        #         train=True
        #     )


class MMDCLAIConvCallbackStep(MMDCLAIScatterplotCallbackStep):
    def __init__(self, save_dir: str, n_samples: int):
        super().__init__(save_dir)

        self.n_samples = n_samples

    def on_validation_batch_end(
        self,
        trainer: pl.trainer.Trainer,
        pl_module: MMDCDLAILitModuleConv,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx < 3:
            debatch = destructure_batch(batch)[: self.n_samples]

            patch_margin = pl_module.margin
            assert isinstance(patch_margin, int)

            pred = pl_module.predict(debatch)
            lai_pred, lai_gt = (
                pred.lai_pred.squeeze(1).nan_to_num(),
                pred.lai_gt.squeeze(1).nan_to_num(),
            )

            logger.info(lai_pred.shape)
            logger.info(lai_gt.shape)

            reg_input = pred.latent

            logger.info(f"{reg_input.shape=}")
            logger.info(f"{lai_gt.shape=}")
            logger.info(f"{lai_pred.shape=}")
            self.lai_gt_show_img(
                lai_pred,
                lai_gt,
                None,
                reg_input,
                trainer.current_epoch,
                pl_module.model_snap.variable_min.cpu(),
                pl_module.model_snap.variable_max.cpu(),
                batch_idx,
                margin=patch_margin,
            )

            flat_gt = lai_gt[
                :, patch_margin:-patch_margin, patch_margin:-patch_margin
            ].reshape(-1)
            flat_pred = lai_pred[
                :, patch_margin:-patch_margin, patch_margin:-patch_margin
            ].reshape(-1)

            self.lai_gt_pred_scatterplots(
                flat_pred[flat_gt != 0],
                flat_gt[flat_gt != 0],
                trainer.current_epoch,
                pl_module.model_snap.variable_min.cpu(),
                pl_module.model_snap.variable_max.cpu(),
                batch_idx,
            )

    def prepare_img_data(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        pos: list[torch.Tensor, torch.Tensor],
        reg_input: torch.Tensor,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
        margin: int = 28,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred_img = preds.cpu().detach()
        gt_img = gt.cpu().detach()
        input_img = reg_input.cpu().detach()

        pred_img = pred_img[:, margin:-margin, margin:-margin].numpy()
        gt_img = gt_img[:, margin:-margin, margin:-margin].numpy()

        input_img = input_img[:, :, margin:-margin, margin:-margin].numpy()

        return pred_img, gt_img, input_img, np.arange(self.n_samples)

    def prepare_data(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        # idx = np.random.choice(len(preds), int(len(preds) * 0.05), replace=False)
        preds = preds.cpu().detach()
        gt = gt.cpu()
        rmse = np.round(self.loss_fn(preds, gt).item(), 2)

        logging.info("pred max " + str(preds.max()))
        logging.info("pred min " + str(preds.min()))

        logging.info("gt max " + str(gt.max()))
        logging.info("gt min " + str(gt.min()))

        return preds.numpy(), gt.numpy(), rmse
