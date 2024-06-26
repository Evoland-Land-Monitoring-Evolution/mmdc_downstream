# Taken from
# https://github.com/VSainteuf/utae-paps/blob/3d83dc50e67e7ae204559e819cfbb432b1b21e10/src/dataset.py#L85
# some adaptations were made
import logging
import os
from collections.abc import Iterable
from datetime import datetime
from typing import Literal

import numpy as np
import torch
import torch.utils.data as tdata
from mmdc_singledate.models.datatypes import VAELatentSpace
from torch import nn
from torch.nn import functional as F

from mmdc_downstream_pastis.constant.pastis import LABEL_PASTIS
from mmdc_downstream_pastis.datamodule.datatypes import (
    BatchInputUTAE,
    PastisFolds,
    PASTISOptions,
)
from mmdc_downstream_pastis.datamodule.pastis_oe import (
    PASTISDataset,
    PastisOEDataModule,
)

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PASTISEncodedDataset(PASTISDataset):
    def __init__(
        self,
        options: PASTISOptions,
        reference_date: str = "2018-09-01",
        sats=["S2"],
        crop_size=64,
        # dict_classes=None,
        crop_type: Literal["Center", "Random"] = "Random",
        transform: nn.Module | None = None,
        norm: bool = False,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.

            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        """

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

        self.crop_size = crop_size
        self.crop_type = crop_type

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.options = options

        self.memory = {}
        self.memory_dates = {}

        self.sats = sats

        # Get metadata
        meta_patch = self.get_metadata()
        self.id_patches = meta_patch.index
        self.len = len(self.id_patches)

        # self.dict_classes = dict_classes if dict_classes is not None else LABEL_PASTIS
        self.dict_classes = LABEL_PASTIS

        self.labels = list(self.dict_classes.values())

        if norm:
            self.norm: dict[
                str, dict[str, tuple[torch.Tensor, torch.Tensor]]
            ] = self.get_stats()
        else:
            self.norm = None

        logger.info("Normalization values")
        logger.info(self.norm)

        logger.info("Done.")
        print("Dataset ready.")

    # def get_stats(self):
    #     stats = {}
    #     for sat in self.sats:
    #         for m in ("mu", "logvar"):
    #             stats = torch.load(self.options.dataset_path_oe,
    #             f"stats_{m}_{sat}.pt")
    #             scale_regul = nn.Threshold(1e-10, 1.0)
    #             shift = stats.median
    #             scale = scale_regul((stats.qmax - stats.qmin) / 2.0)
    #             stats[sat][m] = [shift, scale]
    #     return stats

    def get_stats(self) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        stats_dict = {s: {} for s in self.sats}
        for sat in self.sats:
            for m in ("mu", "logvar"):
                stats = torch.load(
                    os.path.join(self.options.dataset_path_oe, f"stats_{m}_{sat}.pt")
                )
                stats_dict[sat][m] = stats
        return stats_dict

    def read_data_from_disk(
        self,
        item: int,
        id_patch: int,
    ) -> tuple[
        dict[str, VAELatentSpace],
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Get id_patch from disk and cache it as item in the dataset"""
        data_pastis = {
            satellite: self.load_file(
                os.path.join(
                    self.options.dataset_path_oe,
                    satellite,
                    f"{satellite}_{id_patch}.pt",
                )
            )
            for satellite in self.sats
        }  # pastis_eo.dataclass.PASTISItem

        # Retrieve date sequences
        data = {
            s: (a["latents"] if a is not None else None) for s, a in data_pastis.items()
        }
        if self.norm is not None:
            data.update(
                {
                    s: VAELatentSpace(
                        (
                            (a.mean - self.norm[s]["mu"][0][None, :, None, None])
                            / (self.norm[s]["mu"][1][None, :, None, None].nan_to_num(1))
                        ).nan_to_num(),
                        (
                            (a.logvar - self.norm[s]["logvar"][0][None, :, None, None])
                            / (
                                self.norm[s]["logvar"][1][
                                    None, :, None, None
                                ].nan_to_num(1)
                            )
                        ).nan_to_num(),
                    )
                    for s, a in data.items()
                }
            )

        dates = {
            s: (self.prepare_dates(a["dates"]) if a is not None else None)
            for s, a in data_pastis.items()
        }
        masks = {
            s: (a["mask"] if a is not None else None) for s, a in data_pastis.items()
        }

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
                    {k: v for k, v in masks.items()},
                    target,
                    {k: d for k, d in dates.items()},
                ]
            else:
                self.memory[item] = [data, masks, target, dates]

        return data, masks, target, dates

    def get_one_satellite_patches(self, satellite: str) -> np.array:
        """
        For each satellite, we check available patches
        """
        path = self.options.dataset_path_oe
        return [
            int(file[:-3].split("_")[-1])
            for file in os.listdir(os.path.join(path, satellite))
        ]

    def patches_to_keep(self) -> np.array:
        """
        We make lists of all available patches for a satellite.
        In case we use several satellites, we choose patches present for all of them.
        """

        valid_patches = []
        if len(self.sats) == 1:
            return self.get_one_satellite_patches(self.sats[0])

        elif len(self.sats) == 2:
            for satellite in self.sats:
                valid = self.get_one_satellite_patches(satellite)
                valid_patches.append(valid)
            return np.intersect1d(
                valid_patches[0], valid_patches[1], assume_unique=True
            )
        else:
            raise NotImplementedError

    def __getitem__(
        self, item: int
    ) -> (
        dict[str, VAELatentSpace],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        int,
    ):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.options.cache or item not in self.memory.keys():
            data, data_masks, target, dates = self.read_data_from_disk(item, id_patch)

        else:
            data, data_masks, target, dates = self.memory[item]
            if self.options.mem16:
                data = {
                    k: v.casting(torch.float32) for k, v in data.items()
                }  # TODO casting

        t, c, h, w = data[self.sats[0]].mean.shape
        if self.crop_size is not None:
            y, x = self.get_crop_idx(
                rows=h, cols=w
            )  # so that same crop for all the bands of a sits
            # TODO I stopped here get data mask
            target = target[y : y + self.crop_size, x : x + self.crop_size]
            data = {sat: self.clip_vae(data[sat], crop_xy=(x, y)) for sat in self.sats}
            data_masks = {
                sat: data_masks[sat][
                    :, :, y : y + self.crop_size, x : x + self.crop_size
                ]
                for sat in self.sats
            }

        target[target == 19] = 0  # merge background and void class
        mask = (target == 0) | (target == 19)

        return data, data_masks, dates, target, mask, id_patch

    def clip_vae(self, vae: VAELatentSpace, crop_xy: tuple[int, int]) -> VAELatentSpace:
        """
        Crop embeddings
        """
        x, y = crop_xy
        return VAELatentSpace(
            vae.mean[:, :, y : y + self.crop_size, x : x + self.crop_size],
            vae.logvar[:, :, y : y + self.crop_size, x : x + self.crop_size],
        )


def pad_tensor(x_t: torch.Tensor, final_size: int, pad_value: int = 0) -> torch.Tensor:
    """Pad the first dimension of a tensor with pad_value to have final_size"""
    padlen = final_size - x_t.shape[0]
    padding = [0 for _ in range(2 * len(x_t.shape[1:]))] + [0, padlen]
    return F.pad(x_t, pad=padding, value=pad_value)


def pad_collate_vae(
    batch: Iterable[
        dict[str, VAELatentSpace],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        int,
    ],
    use_logvar: bool | dict[str, bool] = True,
) -> BatchInputUTAE:
    batch_dict = {}
    data_masks_dict = {}
    doys_dict = {}
    sat_dict, data_masks, doys, target, mask, id_patch = zip(*batch)
    target = torch.stack(target, 0)
    mask = torch.stack(mask, 0)

    sats = list(sat_dict[0].keys())

    max_len = max([len(value.mean) for s in sat_dict for key, value in s.items()])
    for sat in sats:
        items: list[VAELatentSpace] = [v[sat] for v in sat_dict]
        data_masks_dict[sat] = torch.stack(
            [pad_tensor(v[sat].to(DEVICE), max_len) for v in data_masks]
        )
        doys_dict[sat] = torch.stack(
            [pad_tensor(v[sat].to(DEVICE), max_len) for v in doys]
        )

        logvar_yes = use_logvar if type(use_logvar) is bool else use_logvar[sat]

        if logvar_yes:
            batch_dict[sat] = torch.concat(
                [
                    torch.stack(
                        [pad_tensor(item.mean, max_len).to(DEVICE) for item in items]
                    ),
                    torch.stack(
                        [pad_tensor(item.logvar, max_len).to(DEVICE) for item in items]
                    ),
                ],
                dim=-3,
            )
        else:
            batch_dict[sat] = torch.concat(
                [
                    torch.stack(
                        [pad_tensor(item.mean, max_len).to(DEVICE) for item in items]
                    ),
                ],
                dim=-3,
            )
    if len(sats) == 1:
        return BatchInputUTAE(
            sits=batch_dict[sats[0]],
            doy=doys_dict[sats[0]],
            gt=target,
            sits_mask=data_masks_dict[sats[0]],
            gt_mask=mask,
            id_patch=id_patch,
        )

    return BatchInputUTAE(
        sits=batch_dict,
        doy=doys_dict,
        gt=target,
        sits_mask=data_masks_dict,
        gt_mask=mask,
        id_patch=id_patch,
    )


class PastisEncodedDataModule(PastisOEDataModule):
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
        sats: list[str] = ["S2"],
        reference_date: str = "2018-09-01",
        task: Literal["semantic"] = "semantic",
        batch_size: int = 2,
        crop_size: int | None = 64,
        crop_type: Literal["Center", "Random"] = "Random",
        num_workers: int = 1,
        norm: bool = False,
        use_logvar: bool = True,
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
        self.use_logvar = use_logvar
        self.norm = norm

    def instanciate_dataset(self, fold: list[int] | None) -> PASTISDataset:
        return PASTISEncodedDataset(
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
            collate_fn=lambda x: pad_collate_vae(x, self.use_logvar),
        )


#
# def build_dm(sats) -> PastisEncodedDataModule:
#     """Builds datamodule"""
#     return PastisEncodedDataModule(
#         dataset_path_oe=dataset_path_oe,
#         dataset_path_pastis=dataset_path_pastis,
#         folds=PastisFolds([1, 2, 3], [4], [5]),
#         sats=sats,
#         task="semantic",
#         batch_size=4,
#     )
#
#
# model = "2024-04-05_14-58-22"
# # dataset_path_oe = f"{os.environ['WORK']}/results/Pastis_encoded"
# # dataset_path_pastis = f"{os.environ['SCRATCH']}/scratch_data/Pastis"
# dataset_path_oe = "/home/kalinichevae/jeanzay/results/Pastis_encoded"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"
# dataset_path_oe = os.path.join(dataset_path_oe, model)
#
#
# def pastisds_dataloader(sats) -> None:
#     """Use a dataloader with PASTIS dataset"""
#     dm = build_dm(sats)
#     dm.setup(stage="fit")
#     # dm.setup(stage="test")
#     assert (
#         hasattr(dm, "train_dataloader")
#         and hasattr(dm, "val_dataloader")
#         # and hasattr(dm, "test_dataloader")
#     )  # type: ignore[truthy-function]
#     for loader in (
#         dm.train_dataloader(),
#         dm.val_dataloader(),
#     ):  # , dm.test_dataloader()):
#         assert loader
#         for (batch_dict, mask_dict, doys_dict, target, mask, id_patch), _ in zip(
#             loader, range(4)
#         ):
#             if len(sats) > 1:
#                 assert list(batch_dict.keys()) == sats
#                 for sat in sats:
#                     b_s_x, t_s_x, nb_b, p_s_x, _ = batch_dict[sat].shape
#
#                     b_s_imsk, t_s_imsk, nb_b_imsk, p_s_imsk, _ = mask_dict[sat].shape
#
#                     b_s_y, p_s_y, _ = target.shape
#                     b_s_m, p_s_m, _ = mask.shape
#                     b_s_id = len(id_patch)
#
#                     assert b_s_x == b_s_imsk == b_s_y == b_s_m == b_s_id
#                     assert t_s_x == t_s_imsk
#                     assert p_s_x == p_s_imsk == p_s_y == p_s_m
#             else:
#                 b_s_x, t_s_x, nb_b, p_s_x, _ = batch_dict.shape
#
#                 b_s_imsk, t_s_imsk, nb_b_imsk, p_s_imsk, _ = mask_dict.shape
#
#                 b_s_y, p_s_y, _ = target.shape
#                 b_s_m, p_s_m, _ = mask.shape
#                 b_s_id = len(id_patch)
#
#                 assert b_s_x == b_s_imsk == b_s_y == b_s_m == b_s_id
#                 assert t_s_x == t_s_imsk
#                 assert p_s_x == p_s_imsk == p_s_y == p_s_m
#
#
# pastisds_dataloader(["S1", "S2"])
