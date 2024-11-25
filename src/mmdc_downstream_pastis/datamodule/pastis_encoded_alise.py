# Taken from
# https://github.com/VSainteuf/utae-paps/blob/3d83dc50e67e7ae204559e819cfbb432b1b21e10/src/dataset.py#L85
# some adaptations were made
import logging
import os
from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.utils.data as tdata
from mmdc_singledate.models.datatypes import VAELatentSpace
from torch import nn

from mmdc_downstream_pastis.datamodule.datatypes import (
    BatchInputUTAE,
    PastisFolds,
    PASTISOptions,
)
from mmdc_downstream_pastis.datamodule.pastis_encoded import (
    PastisEncodedDataModule,
    PASTISEncodedDataset,
)
from mmdc_downstream_pastis.datamodule.pastis_oe import PASTISDataset

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PASTISEncodedDatasetAlise(PASTISEncodedDataset):
    def __init__(
        self,
        options: PASTISOptions,
        reference_date,
        sats=["S1_ASC"],
        crop_size=64,
        # dict_classes=None,
        crop_type: Literal["Center", "Random"] = "Random",
        transform: nn.Module | None = None,
        norm: bool = False,
        postfix: str = "",
    ):
        """ """

        super().__init__(
            options,
            reference_date,
            sats,
            crop_size,
            crop_type=crop_type,
            transform=transform,
        )

        if sats is None:
            sats = ["S2"]

        self.sats = sats
        self.postfix = "_" + postfix if postfix else ""

    def read_data_from_disk(
        self,
        item: int,
        id_patch: int,
    ) -> tuple[dict[str, VAELatentSpace], torch.Tensor,]:
        """Get id_patch from disk and cache it as item in the dataset"""
        data_pastis = {
            satellite: self.load_file(
                os.path.join(
                    self.options.dataset_path_oe,
                    satellite + self.postfix,
                    f"{satellite}_{id_patch}.pt",
                )
            )
            for satellite in self.sats
        }  # pastis_eo.dataclass.PASTISItem

        # Retrieve date sequences
        data = {s: (a if a is not None else None) for s, a in data_pastis.items()}

        if self.options.task == "semantic":
            target = np.load(
                os.path.join(
                    self.options.dataset_path_pastis,
                    "ANNOTATIONS",
                    f"TARGET_{id_patch}.npy",
                )
            )
            target = torch.from_numpy(target[0].astype(int))
        else:
            raise NotImplementedError
        if self.options.cache:  # Not sure it works
            if self.options.mem16:
                self.memory[item] = [
                    {
                        k: v.casting(torch.float32) for k, v in data.items()
                    },  # TODO: change casting
                ]
            else:
                self.memory[item] = [data, target]

        return data, target

    def get_one_satellite_patches(self, satellite: str) -> np.array:
        """
        For each satellite, we check available patches
        """
        path = self.options.dataset_path_oe
        return [
            int(file[:-3].split("_")[-1])
            for file in os.listdir(os.path.join(path, satellite))
        ]

    def __getitem__(
        self, item: int
    ) -> (dict[str, VAELatentSpace], torch.Tensor, torch.Tensor, int,):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.options.cache or item not in self.memory.keys():
            data, target = self.read_data_from_disk(item, id_patch)

        else:
            data, target = self.memory[item]
            if self.options.mem16:
                data = {
                    k: v.casting(torch.float32) for k, v in data.items()
                }  # TODO casting

        t, c, h, w = data[self.sats[0]].shape
        if self.crop_size is not None:
            y, x = self.get_crop_idx(
                rows=h, cols=w
            )  # so that same crop for all the bands of a sits
            # TODO I stopped here get data mask
            target = target[y : y + self.crop_size, x : x + self.crop_size]
            data = {sat: self.clip_vae(data[sat], crop_xy=(x, y)) for sat in self.sats}

        target[target == 19] = 0  # merge background and void class
        mask = (target == 0) | (target == 19)

        return data, target, mask, id_patch


def pad_collate_alise(
    batch: Iterable[
        dict[str, VAELatentSpace],
        torch.Tensor,
        torch.Tensor,
        int,
    ],
) -> BatchInputUTAE:
    batch_dict = {}

    sat_dict, target, mask, id_patch = zip(*batch)
    target = torch.stack(target, 0)
    mask = torch.stack(mask, 0)

    sats = list(sat_dict[0].keys())

    for sat in sats:
        items = [v[sat] for v in sat_dict]
        batch_dict[sat] = torch.concat(
            [
                torch.stack([torch.Tensor(item).to(DEVICE) for item in items]),
            ],
            dim=0,
        )
    if len(sats) == 1:
        return BatchInputUTAE(
            sits=batch_dict[sats[0]],
            doy=None,
            gt=target,
            sits_mask=None,
            gt_mask=None,
            id_patch=id_patch,
        )

    return BatchInputUTAE(
        sits=batch_dict,
        doy=None,
        gt=target,
        sits_mask=None,
        gt_mask=None,
        id_patch=id_patch,
    )


class PastisEncodedAliseDataModule(PastisEncodedDataModule):
    """
    A DataModule implements 4 key methods:
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(
        self,
        dataset_path_oe: str,
        dataset_path_pastis: str,
        folds: PastisFolds | None,
        sats: list[str] = ["S1_ASC"],
        reference_date: str | None = None,
        task: Literal["semantic"] = "semantic",
        batch_size: int = 2,
        crop_size: int | None = 64,
        crop_type: Literal["Center", "Random"] = "Random",
        num_workers: int = 1,
        postfix: str = "",
    ):
        super().__init__(
            dataset_path_oe,
            dataset_path_pastis,
            folds,
            sats,
            reference_date,
            task,
            batch_size,
            crop_size=crop_size,
            crop_type=crop_type,
            num_workers=num_workers,
        )

        self.postfix = postfix

    def instanciate_dataset(self, fold: list[int] | None) -> PASTISDataset:
        return PASTISEncodedDatasetAlise(
            PASTISOptions(
                task=self.task,
                folds=fold,
                dataset_path_oe=self.dataset_path_oe,
                dataset_path_pastis=self.dataset_path_pastis,
            ),
            sats=self.sats,
            reference_date=self.reference_date,
            crop_size=self.crop_size,
            crop_type=self.crop_type,
            norm=self.norm,
            postfix=self.postfix,
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
            collate_fn=pad_collate_alise,
        )
