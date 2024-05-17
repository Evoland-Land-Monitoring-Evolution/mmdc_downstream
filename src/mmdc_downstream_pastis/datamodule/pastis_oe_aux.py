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
    batch_dict, target, mask, id_patch = custom_collate_classif(batch, pad_value)
    sats = list(batch_dict.keys())[0]
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
