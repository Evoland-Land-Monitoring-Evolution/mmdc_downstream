import logging
from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as pl
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from openeo_mmdc.dataset.dataclass import Stats
from torch import nn

from mt_ssl.data.classif_class import ClassifBInput
from mt_ssl.data.mt_batch import BOutputReprEnco
from mt_ssl.hydra_config.model import ConfigDecoder, ConfigShallowClassifier
from mt_ssl.model.convolutionalblock import ConvBlock, LastBlock
from mt_ssl.model.gf_repr_encoder import GFReprEncoder
from mt_ssl.model.lq_repr_encoder import LQReprEncoder
from mt_ssl.model.mask_repr_encoder import BERTReprEncoder
from mt_ssl.model.mtan_repr_encoder import MTANReprEncoder
from mt_ssl.model.mtan_repr_encoder_2pe import MTANReprEncoder2Pe
from mt_ssl.model.shallow_classifier import MasterQueryDecoding
from mt_ssl.model.template_repr_encoder import (
    BaseReprEncoder,
    BasicReprEncoder,
)
from mt_ssl.module.dataclass import OutFTForward
from mt_ssl.module.template import TemplateClassifModule
from mt_ssl.module.utils import (
    load_interpol_pretrained_module,
    load_mask_pretrained_module,
)

my_logger = logging.getLogger(__name__)


@dataclass
class FTParams:
    pl_module: None | pl.LightningModule
    no_model: bool
    ckpt_path: str | None
    freeze_representation_encoder: bool
    pretrain_module_config: DictConfig
    d_model: int | None = None

    def to_dict(self):
        return {
            "pl_module": self.pl_module,
            "no_model": self.no_model,
            "ckpt_path": self.ckpt_path,
            "freeze_representation_encoder": (
                self.freeze_representation_encoder
            ),
            "pretrain_module_config": self.pretrain_module_config,
            "d_model": self.d_model,
        }


class FineTuneModule(TemplateClassifModule):
    def __init__(
        self,
        train_config,
        pretraining_type: Literal["bert", "mt", "interpol"],
        ft_params: FTParams | dict,
        input_channels: int,
        num_classes: int,
        decoder_config: DictConfig | ConfigDecoder | ConvBlock,
        decoder_type: Literal["linear", "master_query"] = "linear",
        stats: Stats | None = None,
    ):
        super().__init__(train_config, input_channels, num_classes, stats)
        if isinstance(ft_params, dict):
            ft_params = FTParams(**ft_params)
        self.pretraining_type = pretraining_type
        my_logger.info(f"We are loading {pretraining_type} module")
        if pretraining_type == "bert":
            pretrained_module = load_mask_pretrained_module(
                pl_module=ft_params.pl_module,
                path_ckpt=ft_params.ckpt_path,
                no_model=ft_params.no_model,
                params_module=ft_params.pretrain_module_config,
            )
        elif pretraining_type == "mt":
            raise NotImplementedError
            # pretrained_module = load_mt_pretrained_module(
            #     path_ckpt=ft_params.ckpt_path,
            #     no_model=ft_params.no_model,
            #     params_module=ft_params.pretrain_module_config,
            #     input_channels=input_channels,
            # )
        elif pretraining_type == "interpol":
            pretrained_module = load_interpol_pretrained_module(
                pl_module=ft_params.pl_module,
                path_ckpt=ft_params.ckpt_path,
                no_model=ft_params.no_model,
                params_module=ft_params.pretrain_module_config,
                input_channels=input_channels,
            )
        else:
            raise NotImplementedError
        self.freeze_repr_encoder = ft_params.freeze_representation_encoder
        if pretrained_module is not None:
            if ft_params.freeze_representation_encoder:
                pretrained_module.freeze()
                self.repr_encoder: BaseReprEncoder = (
                    pretrained_module.repr_encoder
                )
                self.repr_encoder.eval()
                my_logger.info("Deactivate repr encoder BN and DO layer")
            else:
                self.repr_encoder: BaseReprEncoder = (
                    pretrained_module.repr_encoder
                )
                my_logger.info("We are not freezing the repr encoder ")
            self.d_model = pretrained_module.repr_encoder.d_model
        else:
            my_logger.info("No repr encoder")
            # self.repr_encoder = None
            assert FTParams.d_model is not None, (
                "If no pretrained model given, d_model should at least be"
                " indicated "
            )
            self.d_model = FTParams.d_model
            self.repr_encoder: BaseReprEncoder = BasicReprEncoder(
                d_model=self.d_model, input_channels=input_channels
            )  # a basic linear layer
        if isinstance(decoder_config, nn.Module):
            self.shallow_classifier = decoder_config
        elif isinstance(
            decoder_config,
            DictConfig | ConfigDecoder | ConfigShallowClassifier,
        ):
            if isinstance(self.repr_encoder, LQReprEncoder):
                len_ref_time_points = self.repr_encoder.temp_proj.n_q
            elif isinstance(
                self.repr_encoder,
                (MTANReprEncoder2Pe, MTANReprEncoder, GFReprEncoder),
            ):
                len_ref_time_points = len(
                    self.repr_encoder.reference_time_points
                )
            else:
                len_ref_time_points = None
            if decoder_type == "linear":
                self.shallow_classifier: LastBlock | MasterQueryDecoding = (
                    instantiate(
                        decoder_config,
                        inplanes=self.d_model * len_ref_time_points,
                        planes=num_classes,
                    )
                )
            elif decoder_type == "master_query":
                self.shallow_classifier: LastBlock | MasterQueryDecoding = (
                    instantiate(
                        decoder_config,
                        inplanes=self.d_model,
                        planes=num_classes,
                    )
                )
            else:
                raise NotImplementedError
        my_logger.info(
            f"The shallow classifier is : {type(self.shallow_classifier)}"
        )
        self.save_hyperparameters(ignore=["ft_params"])
        self.ft_params = ft_params

    def forward_representation_encoder(
        self, batch: ClassifBInput, return_attns: bool = False
    ) -> BOutputReprEnco:
        input_sits = batch.sits
        my_logger.debug(batch.padd_index.shape)
        my_logger.debug(input_sits.shape)
        # print("begin {} {}".format(input_sits.shape, input_doy))

        out_repr = self.repr_encoder.forward_keep_input_dim(batch)

        return out_repr

    def forward(
        self, batch: ClassifBInput, return_attns: bool = False
    ) -> OutFTForward:
        if self.freeze_repr_encoder:
            self.repr_encoder.eval()
            with torch.no_grad():
                out_repr = self.forward_representation_encoder(
                    batch, return_attns=return_attns
                )
        else:
            out_repr = self.forward_representation_encoder(
                batch, return_attns=return_attns
            )

        if isinstance(self.shallow_classifier, MasterQueryDecoding):
            if isinstance(self.repr_encoder, BERTReprEncoder):
                padd_index = batch.padd_index
            else:
                padd_index = None
            out_class = self.shallow_classifier(
                out_repr.repr, key_padding_mask=padd_index
            )  # out dim is b h w nc
        else:
            repr = rearrange(
                out_repr.repr, " b k c h w -> b h w (k c)"
            )  # features are fetures and dates
            out_class = self.shallow_classifier(repr)  # out dim is b h w nc

        return OutFTForward(out_class, out_repr.repr)
