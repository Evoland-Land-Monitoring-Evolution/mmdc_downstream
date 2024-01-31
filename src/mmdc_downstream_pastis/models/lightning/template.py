import logging
from abc import abstractmethod

import lightning.pytorch as pl
import pandas as pd
import torch
from einops import rearrange
from hydra.utils import instantiate
from openeo_mmdc.dataset.dataclass import Stats
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection

from mt_ssl.data.classif_class import ClassifBInput
from mt_ssl.module.dataclass import OutFTForward, OutFTSharedStep

my_logger = logging.getLogger(__name__)


class TemplateModule(pl.LightningModule):
    def __init__(self, train_config, stats: None | Stats = None):
        super().__init__()

        self.return_attns = train_config.return_attns
        self.train_config = train_config
        self.learning_rate = train_config.lr
        # self.save_hyperparameters(ignore=["train_config", "datamodule"])
        self.scheduler = train_config.scheduler
        self.optimizer = self.train_config.optimizer
        self.bs = train_config.batch_size
        self.stats = stats
        self.metric_name = []

    #        self.train_loss = instantiate(train_config.loss)

    def setup(self, stage: str | None):
        self.border = 0

    # self.example_input_array =next(iter(datamodule.val_dataloader()))
    def configure_optimizers(self):
        optimizer = instantiate(
            self.optimizer, params=self.parameters(), lr=self.learning_rate
        )
        sch = instantiate(self.scheduler, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": self.train_config.optimizer_monitor,
                "strict": False,
            },
        }

    def on_test_epoch_start(self):
        self.df_metrics = pd.DataFrame(
            columns=self.metric_name + ["test_loss"]
        )


class TemplateClassifModule(TemplateModule):
    def __init__(
        self,
        train_config,
        input_channels: int,
        num_classes: int,
        stats: None | Stats = None,
    ):
        super().__init__(train_config, stats=stats)
        my_logger.info(torch.cuda.memory_allocated())
        self.input_channels = input_channels
        self.weights = None
        self.num_classes = num_classes
        self.mask_loss = train_config.mask_loss
        if train_config.metrics is not None:
            my_logger.info(train_config.metrics)
            d_metrics = {}
            for name, cb_conf in train_config.metrics.items():
                one_metric: Metric = instantiate(
                    cb_conf, num_classes=num_classes
                )
                d_metrics.update({name: one_metric})
                metrics = MetricCollection(d_metrics)
                self.train_metrics = metrics.clone(prefix="train_")
                self.val_metrics = metrics.clone(prefix="val_")
                self.test_metrics = metrics.clone(prefix="test_")
                self.save_test_metrics = {}

        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
            self.save_test_metrics = {}
        self.train_loss = instantiate(
            train_config.loss
        )  # , weight=self.weights, _convert_="partial"
        self.val_loss = instantiate(train_config.loss)
        self.test_loss = instantiate(train_config.loss)

    def border_effect(self):
        """How much pixels are disturbed by border values at each convolution"""
        return 0

    @abstractmethod
    def forward(
        self, batch: ClassifBInput, return_attns: bool = False
    ) -> OutFTForward:
        pass

    def shared_step(
        self,
        batch: ClassifBInput,
        batch_idx: int,
        loss_fun: nn.Module | None = None,
    ) -> OutFTSharedStep:
        if loss_fun is None:
            loss_fun = self.train_loss
        out = self.forward(batch)
        # loss=self.train_loss()
        _, h, w = batch.labels.shape
        cropped_prediction = out.seg_map[
            :,
            self.border : -self.border + w,
            self.border : -self.border + w,
            :,
        ]
        cropped_trg = batch.labels[
            :,
            self.border : -self.border + w,
            self.border : -self.border + w,
        ].to(int)
        pred_flatten = rearrange(cropped_prediction, " b h w c -> (b h w) c")
        trg_flatten = rearrange(cropped_trg, "b h w -> ( b h w )")
        kwargs_out = {}
        if self.mask_loss:
            cropped_mask = batch.mask.bool()[
                :,
                self.border : -self.border + w,
                self.border : -self.border + w,
            ]
            mask = rearrange(cropped_mask, "b h w -> (b h w)")
            kwargs_out.update({"mask_loss": cropped_mask})
            if cropped_mask.sum() == 0:
                my_logger.debug("all elem are masked")
                loss = torch.empty(1)
                loss[0] = float("nan")
            else:
                # my_logger.info("pred_flatten shape {}".format(pred_flatten.shape))
                loss = loss_fun(pred_flatten[mask], trg_flatten[mask])
        else:
            loss = loss_fun(pred_flatten, trg_flatten)
        return OutFTSharedStep(
            loss=loss,
            pred=cropped_prediction,
            trg=cropped_trg,
            pred_flatten=pred_flatten,
            trg_flatten=trg_flatten,
            **kwargs_out,
        )

    def training_step(self, batch: ClassifBInput, batch_idx):
        # my_logger.info("train step start {}".format(torch.cuda.memory_allocated()))
        out_shared_step = self.shared_step(
            batch, batch_idx, loss_fun=self.train_loss
        )
        loss = out_shared_step.loss
        if not torch.isnan(out_shared_step.loss):
            metrics = self.apply_metrics(out_shared_step, self.train_metrics)
            metrics["train_loss"] = loss.item()

            self.log_dict(
                metrics,
                on_epoch=True,
                on_step=True,
                batch_size=self.bs,
                prog_bar=True,
            )
            # my_logger.info("train step end {}".format(torch.cuda.memory_allocated()))
            return loss

        return None

    def apply_metrics(self, out_shared_step: OutFTSharedStep, metrics) -> dict:
        if metrics is None:
            return {}
        else:
            if out_shared_step.mask_loss.sum() == 0:
                return {}

            return metrics(
                out_shared_step.pred_flatten.detach(),
                out_shared_step.trg_flatten.detach(),
            )

    def validation_step(
        self, batch, batch_idx
    ) -> tuple[dict, OutFTSharedStep] | None:
        # my_logger.info("Validation step start {}".format(torch.cuda.memory_allocated()))
        out_shared_step = self.shared_step(
            batch, batch_idx, loss_fun=self.val_loss
        )
        loss = out_shared_step.loss
        metrics = self.apply_metrics(out_shared_step, self.val_metrics)

        if not torch.isnan(out_shared_step.loss):
            metrics["val_loss"] = loss.item()
            self.log_dict(
                metrics,
                on_epoch=True,
                on_step=True,
                batch_size=self.bs,
                prog_bar=True,
            )
            # my_logger.info("Log loss{}".format(metrics["val_loss"]))

        # my_logger.info("Validation step end {}".format(torch.cuda.memory_allocated()))
        return metrics, out_shared_step

    def on_test_epoch_start(self):
        self.df_metrics = pd.DataFrame(
            columns=self.metric_name + ["test_loss"]
        )

    def test_step(
        self, batch, batch_idx
    ) -> tuple[Tensor, OutFTSharedStep] | None:
        out_shared_step = self.shared_step(
            batch, batch_idx, loss_fun=self.train_loss
        )
        loss = out_shared_step.loss
        metrics = {}
        # df = pd.DataFrame(columns=self.metric_name + ["test_loss"])
        if not torch.isnan(out_shared_step.loss):
            metrics["test_loss"] = torch.Tensor([loss.item()])
            r_pred = out_shared_step.pred_flatten.detach()
            r_trg = out_shared_step.trg_flatten.detach()
            my_loss = loss.item()
            self.test_metrics.update(r_pred, r_trg)
            # my_logger.info("Metrics in test {}".format(metrics))

        else:
            my_loss = None
        out_shared_step.loss = my_loss
        return my_loss, out_shared_step

    def on_test_epoch_end(self) -> None:
        outputs = self.test_metrics.compute()
        outputs = {
            k: v.to(device="cpu", non_blocking=True)
            for k, v in outputs.items()
        }
        self.save_test_metrics = outputs
