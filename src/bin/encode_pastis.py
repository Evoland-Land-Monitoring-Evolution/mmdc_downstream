#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

from mmdc_downstream_pastis.datamodule.pastis_oe import PastisDataModule

dataset_path_oe = "/work/CESBIO/projects/DeepChange/Ekaterina/Pastis_OE"
dataset_path_pastis = "/work/CESBIO/projects/DeepChange/Iris/PASTIS"


def build_dm(sats) -> PastisDataModule:
    """Builds datamodule"""
    return PastisDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=None,
        sats=sats,
        task="semantic",
        batch_size=1,
        max_len=0,
    )


sats = ["S2", "S1_ASC", "S1_DESC"]
dm = build_dm(sats)

dataset = dm.instanciate_dataset(fold=None)
loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
for batch_dict, target, mask, id_patch in loader:
    print(id_patch)
    for satellite in batch_dict.keys():
        batch_sat = batch_dict[satellite]
        input_doy = batch_sat.input_doy
        padd_val = batch_sat.padd_val
        padd_index = ~batch_sat.padd_index
        true_doy = batch_sat.true_doy
        data = batch_sat.sits.data
        x = data.img
        angles = data.angles
        mask = data.mask
        meteo = batch_sat.sits.meteo
        dem = batch_sat.sits.dem
