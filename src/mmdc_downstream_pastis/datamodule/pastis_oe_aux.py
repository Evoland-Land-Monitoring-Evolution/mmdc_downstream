# Taken from
# https://github.com/VSainteuf/utae-paps/blob/3d83dc50e67e7ae204559e819cfbb432b1b21e10/src/dataset.py#L85
# some adaptations were made
import logging
from collections.abc import Iterable
from typing import Literal

import torch
import torch.utils.data as tdata

from mmdc_downstream_pastis.datamodule.datatypes import (
    BatchInputUTAE,
    OneSatellitePatch,
    PastisFolds,
)
from mmdc_downstream_pastis.datamodule.pastis_oe import (
    PastisOEDataModule,
    custom_collate_classif,
)

# from mmdc_downstream_pastis.models.lightning.pastis_utae_semantic import (
#     set_aux_stats,
#     standardize_data,
# )

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def custom_collate_classif_aux(
    batch: Iterable[dict[str, OneSatellitePatch], torch.Tensor, torch.Tensor, int],
    pad_value: int = 0,
) -> BatchInputUTAE:
    """
    Collate batch items and concat aux data to the image data:
    img + ang + meteo + dem
    """
    batch: BatchInputUTAE = custom_collate_classif(batch, pad_value)
    batch_dict, target, mask, id_patch = (
        batch.sits,
        batch.gt,
        batch.gt_mask,
        batch.id_patch,
    )
    sats = list(batch_dict.keys())
    batch_dict_aux = {
        sat: v.concat_aux_data(pad_value) for sat, v in batch_dict.items()
    }
    if len(sats) == 1:
        return BatchInputUTAE(
            sits=batch_dict_aux[sats[0]],
            doy=batch_dict[sats[0]].true_doy,
            gt=target,
            sits_mask=batch_dict[sats[0]].sits.data.mask,
            gt_mask=mask,
            id_patch=id_patch,
        )

    return BatchInputUTAE(
        sits=batch_dict_aux,
        doy={sat: v.true_doy for sat, v in batch_dict.items()},
        gt=target,
        sits_mask={sat: v.sits.data.mask for sat, v in batch_dict.items()},
        gt_mask=mask,
        id_patch=id_patch,
    )


class PastisAuxDataModule(PastisOEDataModule):
    """
    A DataModule to load Patis SITS:
    (images concatenated with angles, meteo and DEM)
    """

    def __init__(
        self,
        dataset_path_oe: str,
        dataset_path_pastis: str,
        folds: PastisFolds | None,
        sats: list[str] = ["S2"],
        reference_date: str = "2018-09-01",
        task: Literal["semantic"] = "semantic",
        batch_size: int = 10,
        strict_s1: bool = True,
        crop_size: int | None = None,
        crop_type: Literal["Center", "Random"] = "Random",
        num_workers: int = 1,
        pad_value: int = 0,
    ):
        super().__init__(
            dataset_path_oe,
            dataset_path_pastis,
            folds,
            sats,
            reference_date,
            task,
            batch_size,
            strict_s1,
            crop_size,
            crop_type,
            num_workers,
            pad_value,
        )

    def instanciate_data_loader(
        self, dataset: tdata.Dataset, shuffle: bool = False, drop_last: bool = True
    ) -> tdata.DataLoader:
        """Return a data loader with the PASTIS data set"""

        return tdata.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: custom_collate_classif_aux(
                x, pad_value=self.pad_value
            ),
        )


#
#
# def build_dm(sats) -> PastisAuxDataModule:
#     """Builds datamodule"""
#     return PastisAuxDataModule(
#         dataset_path_oe=dataset_path_oe,
#         dataset_path_pastis=dataset_path_pastis,
#         folds=PastisFolds([1, 2, 3], [4], [5]),
#         sats=sats,
#         task="semantic",
#         batch_size=1,
#     )
#
#
# dataset_path_oe = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"
# stats_aux_path =
# "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE/stats_S1_ASC.pt"
#
# def pastisds_dataloader(sats) -> None:
#     """Use a dataloader with PASTIS dataset"""
#     dm = build_dm(sats)
#     dm.setup(stage="fit")
#     dm.setup(stage="test")
#     stats = set_aux_stats(
#                 stats_aux_path,
#                 device="cpu",
#                 sat=dm.sats[0]
#                 if type(stats_aux_path) is not dict
#                 else None,
#             )
#     assert (
#         hasattr(dm, "train_dataloader")
#         and hasattr(dm, "val_dataloader")
#         and hasattr(dm, "test_dataloader")
#     )  # type: ignore[truthy-function]
#     for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
#         assert loader
#         for batch, _ in zip(loader, range(4)):
#             sits = batch.sits
#
#             if type(sits) is dict:
#                 sits = {
#                     sat: standardize_data(
#                         sits[sat].clone(),
#                         shift=stats[sat].shift,
#                         scale=stats[sat].scale,
#                         pad_index=batch.doy[sat] == dm.pad_value,
#                         pad_value=dm.pad_value,
#                         mask=batch.sits_mask
#                     )
#                     for sat in sits
#                 }
#             else:
#                 sits = standardize_data(
#                     sits.clone(),
#                     shift=stats.shift,
#                     scale=stats.scale,
#                     pad_index=batch.doy == dm.pad_value,
#                     pad_value=dm.pad_value,
#                     mask=batch.sits_mask
#                 )
#
# pastisds_dataloader(["S1_ASC"])
