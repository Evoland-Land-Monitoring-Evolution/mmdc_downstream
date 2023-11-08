"""Lightning Data Module, torch Dataset and Dataloader for the MMDC
single date data"""
import glob
import logging
import random
from collections.abc import Generator, Iterable, Iterator
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .components.datamodule_utils import average_stats, create_tensors_path_set
from .constants import MMDC_DATA
from .datatypes import (
    MMDCBatch,
    MMDCDataLoaderConfig,
    MMDCDataModuleFiles,
    MMDCDataPaths,
    MMDCDataSets,
    MMDCDataStats,
    MMDCDataStatsConfig,
    MMDCDataStruct,
    MMDCMeteoStats,
    MMDCTensorStats,
)

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# get off the otb - gdal - proj warning
# warnings.filterwarnings("ignore", category=rio.errors.EnvError)


class MMDCDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU,
        not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(
        self,
        data_paths: MMDCDataPaths,
        dl_config: MMDCDataLoaderConfig,
        stats_config: MMDCDataStatsConfig,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        # init the path related variables
        self.data_paths = data_paths
        self.stats_config = stats_config

        # create dataset related variables
        self.dl_config = dl_config

        self.data: MMDCDataSets | None = None

        # stats
        self.stats: MMDCDataStats | None = None

        self.files = MMDCDataModuleFiles([], [], [], [])
        self.available_tiles: list[str] | None = None

    def prepare_data(self) -> None:
        """
        Verify that the tensors already exist

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """

        self.find_available_tiles_and_rois()

    def build_stats_files_names(self) -> None:
        """Find the files containing the stats for the data set and
        update the object state"""
        assert isinstance(self.files.train_data_files, list)
        logger.debug("Train data files %s", self.files.train_data_files)
        for data_set in self.files.train_data_files:
            set_list = []
            logger.debug("Vars in data set %s", data_set)
            for var in data_set:
                path = Path(str(var))
                set_dir = path.parent
                filename = path.name
                if (  # we don't compute stats on s1/s2 angles, masks or
                    # existing stats
                    "angles" in filename
                    or "mask" in filename
                    or "stats" in filename
                ):
                    logger.debug("Continue for %s", var)
                    continue

                parts = filename.split("_")
                suffix = parts[-1]
                tile = parts[0]
                set_name = "_".join(parts[1:-1]).replace("_set", "")
                stats_set_name = f"{set_dir}/{tile}_stats_{set_name}_{suffix}"
                logger.debug("Stats set name %s", stats_set_name)
                set_list.append(stats_set_name)
            self.files.stats_files.append(set_list)
        assert self.files.stats_files
        logger.debug("Stats files = %s", self.files.stats_files)

    def check_rois_exist(self, roi_file: Path) -> bool:
        """Verify that the ROIs in the ROI file really exist on disk"""
        with roi_file.open() as rois:
            for tile_roi in rois:
                if tile_roi.strip():
                    logger.info("Tile, roi: %s", tile_roi)
                    tile, roi = tile_roi.strip().split("_")
                    roi_regex = (
                        f"{self.data_paths.tensors_dir}/{tile}/"
                        f"{tile}_s1_set_*_{roi}.pth"
                    )
                    if not glob.glob(roi_regex):
                        logger.info("Missing ROI: %s \n %s", tile_roi, roi_regex)
                        return False
        return True

    def find_available_tiles_and_rois(self) -> None:
        """Update the list of available tiles and rois wrt what is
        available on disk"""
        assert (
            self.data_paths.tensors_dir.exists()
        ), f"{self.data_paths.tensors_dir} does not exist"
        self.available_tiles = [
            Path(t).name
            for t in glob.glob(f"{self.data_paths.tensors_dir}/{('[0-9]' * 2)}*")
        ]
        assert self.available_tiles, f"No data found in {self.data_paths.tensors_dir}"
        logger.info("Available tiles: %s ", self.available_tiles)
        logger.info("Requested train ROIS: %s", self.data_paths.train_rois)
        logger.info("Requested val ROIS: %s", self.data_paths.val_rois)
        logger.info("Requested test ROIS: %s", self.data_paths.test_rois)

        assert self.check_rois_exist(self.data_paths.train_rois)
        assert self.check_rois_exist(self.data_paths.val_rois)
        assert self.check_rois_exist(self.data_paths.test_rois)

    def build_file_list(self, roi_file: Path) -> list:
        """Return the list of files corresponding to the ROIs listed in
        the roi_file"""
        logger.info("ROI file: %s", roi_file)
        res = []
        with roi_file.open() as rois:
            for tile_roi in rois:
                tile, roi = tile_roi.split("_")
                file_set = [
                    list(i)
                    for i in create_tensors_path_set(
                        "",
                        path_to_exported_files=self.data_paths.tensors_dir,
                        tile_list=[tile],
                        roi_list=[int(roi)],
                    )
                ]
                res.extend(file_set)
        return res

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables:
        `self.data.train`, `self.data.val`, `self.data.test`.

        This method is called by lightning when
        doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice!
        The `stage` can be used to differentiate whether
        it's called before trainer.fit()` or `trainer.test()`.
        """

        self.files.train_data_files = self.build_file_list(self.data_paths.train_rois)
        self.files.val_data_files = self.build_file_list(self.data_paths.val_rois)
        self.files.test_data_files = self.build_file_list(self.data_paths.test_rois)

        if stage == "fit":
            logger.info(
                " train files : %s",
                [Path(str(ds[0])).name for ds in self.files.train_data_files],
            )
            logger.info(
                " val files : %s",
                [Path(str(ds[0])).name for ds in self.files.val_data_files],
            )
            logger.info(
                " test files : %s",
                [Path(str(ds[0])).name for ds in self.files.test_data_files],
            )

            self.data = MMDCDataSets(
                IterableMMDCDataset(
                    files=self.files.train_data_files,
                    batch_size=self.dl_config.batch_size,
                    max_open_files=self.dl_config.max_open_files,
                ),
                IterableMMDCDataset(
                    files=self.files.val_data_files,
                    batch_size=self.dl_config.batch_size,
                    max_open_files=self.dl_config.max_open_files,
                ),
            )

        if stage == "test":
            assert self.data
            self.data = MMDCDataSets(
                self.data.train,
                self.data.val,
                IterableMMDCDataset(
                    files=self.files.test_data_files,
                    batch_size=self.dl_config.batch_size,
                    max_open_files=self.dl_config.max_open_files,
                ),
            )

    def get_stats_from_torchfiles(
        self,
    ) -> None:
        """Read the dataset statistics from the files for each ROI and average them"""
        logger.info("Reading stats")

        self.build_stats_files_names()

        stats_s2: list[MMDCTensorStats] = []
        stats_s1: list[MMDCTensorStats] = []
        stats_meteo: dict[str, MMDCTensorStats] = {}
        stats_dem: list[MMDCTensorStats] = []
        stats_meteo = {
            k: [] for k in MMDCMeteoStats.__dict__["__dataclass_fields__"].keys()
        }
        # TODO : modify the iteration process with a generator pipeline
        # pylint: disable=possibly-unused-variable
        for (
            stats_s2_reflectances,
            stats_s1_backscatter,
            stats_meteo_dew_temp,
            stats_meteo_prec,
            stats_meteo_sol_rad,
            stats_meteo_temp_max,
            stats_meteo_temp_mean,
            stats_meteo_temp_min,
            stats_meteo_vap_press,
            stats_meteo_wind_speed,
            stats_dem_data,
        ) in self.files.stats_files:
            stats_s2.append(torch.load(stats_s2_reflectances))
            stats_s1.append(torch.load(stats_s1_backscatter))

            for key, value in stats_meteo.items():
                stats_meteo.setdefault(key, value).append(
                    torch.load(locals()["stats_meteo_" + key])
                )

            stats_dem.append(torch.load(stats_dem_data))

        self.stats = MMDCDataStats(
            average_stats(stats_s2),
            average_stats(stats_s1),
            MMDCMeteoStats(
                *[
                    average_stats(stats_meteo[m])
                    for m in MMDCMeteoStats.__dict__["__dataclass_fields__"].keys()
                ]
            ),
            average_stats(stats_dem),
        )
        logger.info("Stats values :")
        logger.info("Stats S2 %s\n", self.stats.sen2)
        logger.info("Stats S1 %s\n", self.stats.sen1)
        logger.info("Stats Meteo %s\n", self.stats.meteo.concat_stats())
        logger.info("Stats DEM %s\n", self.stats.dem)

    def get_stats(self) -> MMDCDataStats:
        """
        Compute stats
        """

        self.get_stats_from_torchfiles()

        assert self.stats is not None
        assert self.stats.sen2 is not None
        assert self.stats.sen1 is not None
        assert self.stats.meteo is not None
        assert self.stats.dem is not None

        return self.stats

    def train_dataloader(self) -> DataLoader:
        assert self.data is not None
        assert self.data.train is not None
        return DataLoader(
            dataset=self.data.train,
            batch_size=None,
            num_workers=self.dl_config.num_workers,
            pin_memory=self.dl_config.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.data is not None
        assert self.data.val is not None
        return DataLoader(
            dataset=self.data.val,
            batch_size=None,
            num_workers=self.dl_config.num_workers,
            pin_memory=self.dl_config.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.data is not None
        assert self.data.test is not None
        return DataLoader(
            dataset=self.data.test,
            batch_size=None,
            num_workers=self.dl_config.num_workers,
            pin_memory=self.dl_config.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def nan_check(data: dict[str, torch.Tensor]) -> None:
    """Sanity check data for Nans presence outside of the mask"""
    for t_name, t_tens in data.items():
        if t_name == "s2_set":
            t_tens = t_tens[
                ~data["s2_masks"].repeat_interleave(t_tens.shape[-3], -3).bool()
            ]
        if t_name == "s2_angles":
            t_tens = t_tens[
                ~data["s2_masks"].repeat_interleave(t_tens.shape[-3], -3).bool()
            ]
        if t_name == "s1_set":
            t_tens = t_tens[~data["s1_valmasks"].repeat_interleave(3, -3).bool()]
        if t_name == "s1_angles":
            t_tens = t_tens[~data["s1_valmasks"].bool()]
        assert (
            t_tens.isnan().sum() == 0
        ), f"NaN detected in tensor {t_name} with shape {t_tens.shape}"


class IterableMMDCDataset(IterableDataset):
    """
    Implements a IterableDataset object
    that allow to read data from serialized
    pytorch files from disk bigger that RAM memory
    """

    def __init__(self, files: Iterable, batch_size: int = 1, max_open_files: int = 4):
        """IterableMMDCDataset Constructor"""

        # Store Variables
        self.files = [list(i) for i in files]
        self.batch_size = batch_size

        self.max_open_files = max_open_files
        self.first_file = 0
        self.iterator: Generator | None = None
        self.created = False
        self.len_: int | None = None

    def shuffle_files(self) -> None:
        """Randomly reorder the files"""
        self.files = random.sample(self.files, len(self.files))

    def load_files(self, file_list: list[list[str]]) -> MMDCDataStruct:
        """Read all files at once, shuffle the sample order and yield the items"""

        # get back the original type of data
        file_zip = (tuple(i) for i in file_list)
        # unzip the variables
        file_unzip = list(zip(*file_zip))

        worker_info = get_worker_info()
        assert worker_info is not None
        w_id = worker_info.id

        keys = tuple(MMDC_DATA)
        logger.info(
            "\nWorker %s will load files in %s \n", w_id, list(tuple(file_unzip)[0])
        )
        data = {
            k: torch.concat([torch.load(f) for f in list(v)])
            for k, v in zip(keys, tuple(file_unzip))
        }

        # create variable permutation
        nb_patches = data["s2_set"].shape[0]
        idx = torch.randperm(nb_patches)

        logger.info("\nWorker %s loaded %s patches in RAM\n", w_id, nb_patches)

        # import time
        # start_time = time.time()
        # nan_check(data)
        # print("--- %s seconds ---" % (time.time() - start_time))

        data["s1_angles"] = data["s1_angles"].nan_to_num()
        data["s2_angles"] = data["s2_angles"].nan_to_num()

        data = MMDCDataStruct.init_empty().fill_empty_from_dict(dictionary=data)

        return data[idx]

    def slice_files(self) -> list[list[list[str]]]:
        """Slice the list of files with max_open_files"""
        # A slice for the file list contains max_open_files
        tmp_fl = list(self.files) + list(self.files)
        return [
            tmp_fl[idx : idx + self.max_open_files]
            for idx in range(0, len(self.files), self.max_open_files)
        ]

    def generate_sample(self) -> Generator:
        """Generate one sample of the data set"""
        self.shuffle_files()
        for files in cycle(self.slice_files()):
            tensors = self.load_files(files)
            yield from zip(
                tensors.s2_data.s2_set,
                tensors.s2_data.s2_masks,
                tensors.s2_data.s2_angles,
                tensors.s1_data.s1_set,
                tensors.s1_data.s1_valmasks,
                tensors.s1_data.s1_angles,
                tensors.meteo.concat_data(),
                tensors.dem,
            )

    def generate_batch(self) -> zip:
        """Generate a batch of samples"""
        assert self.iterator is not None
        return zip(*[self.iterator for _ in range(self.batch_size)])

    def __iter__(self) -> Iterator:
        return self.generate_batch()

    def __len__(self) -> int:
        if self.len_ is None:
            self.len_ = 0
            for file_set in self.files:
                for data_file in file_set:
                    if "s1_angles" in data_file:
                        self.len_ += torch.load(data_file).shape[0]
        elif self.len_ == 0:
            raise RuntimeError("Could not estimate the length of the dataset")
        assert self.len_ is not None
        return self.len_ // self.batch_size

    def __getitem__(self, index: int) -> None:
        """This method should not be called, but is implemented to silent pylint"""


def worker_init_fn(_: Any) -> None:
    """Function used by the dataloader to create a different data set
    for earch worker

    """
    w_info = get_worker_info()
    assert w_info is not None
    dataset = w_info.dataset
    w_id = w_info.id

    if isinstance(dataset, IterableMMDCDataset):
        if not dataset.created:
            dataset.created = True
            split_size = len(dataset.files) // w_info.num_workers
            if split_size == 0:
                raise RuntimeError(
                    f"Number of workers ({w_info.num_workers}) > "
                    f"number of data files ({len(dataset.files)})"
                )
            dataset.files = random.sample(
                dataset.files[w_id * split_size : (w_id + 1) * split_size], split_size
            )
            dataset.max_open_files = dataset.max_open_files // w_info.num_workers
            dataset.iterator = iter(dataset.generate_sample())


def destructure_batch(
    batch: Any,
) -> MMDCBatch:
    """Unpack what we get from the dataloader to a named tuple of tensors"""
    list_mmdc_variables = [
        "s2_reflectances",
        "s2_masks",
        "s2_angles",
        "s1_backscatter",
        "s1_valmasks",
        "s1_angles",
        "meteo_data",
        "dem_data",
    ]
    zip_batch = zip(*batch)

    unpacked_batch = {
        i: torch.stack(j) for (i, j) in zip(list_mmdc_variables, zip_batch)
    }

    s2_x = unpacked_batch["s2_reflectances"]
    s2_m = unpacked_batch["s2_masks"]
    s2_a = unpacked_batch["s2_angles"]
    s1_x = unpacked_batch["s1_backscatter"]
    s1_vm = unpacked_batch["s1_valmasks"]
    s1_a = unpacked_batch["s1_angles"]
    meteo_x = torch.flatten(unpacked_batch["meteo_data"], start_dim=1, end_dim=2)
    dem_x = unpacked_batch["dem_data"]

    return MMDCBatch(
        s2_x=s2_x,
        s2_m=s2_m,
        s2_a=s2_a,
        s1_x=s1_x,
        s1_vm=s1_vm,
        s1_a=s1_a,
        meteo_x=meteo_x,
        dem_x=dem_x,
    )
