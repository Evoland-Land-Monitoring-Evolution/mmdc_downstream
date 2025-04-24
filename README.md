## Overview

This code is designed to validate the quality of image embeddings on different downstream tasks (DST).

We test the images produced by the following embedding algorithms:

- MMDC
- ALISE
- MALICE
- MALICE Aux
- pVAE

The embeddings were evaluated on the following DST tasks:

- Leaf-area-Index (LAI) regression
- Tree Cover Density (TCD) regression
- PASTIS crop classification
- BioMass and Tree Height regression

## What's inside?

This repository contains multiple scripts to:

- compute and save the embeddings with different algorithms,
- perform DST with those embeddings.

Note that we do not compute the embeddings "on the go" while training DST algorithms, but always pre-compute that embeddings.

## Downstream Tasks

### LAI

### TCD

To assess the quality of our embeddings, we compare them to the original benchmark VLCC 2018 TCD of GAF partners.
For the raw Sentinel-2 data, we choose all the images available from March to October 2018 (leaf season) with cloud coverage less than 60\%, plus 4 less cloudy images per month. However, contrary to the original benchmark, we equally assess Sentinel-1 data performance, exploiting all the available images for the given timestamp.
For the selected dates, we compute the feature-wise median values for each month to generate a monthly synthesis. Finally the processed data is passed to the CatBoost Regression model.

While GAF partners use the following hyperparameters to train their model:
iterations=150, l2_leaf_reg=17, depth=7, we have empirically chosen the following parameters:
iterations=750, l2_leaf_reg=17, depth=10.

We use the same date selection strategy for our single-date embedding algorithms (MMDC, pVAE).
However, as the multi-temporal embedding algorithms (ALISE, MALICE, MALICE Aux) already produce the cloud-free temporal synthesis, there is no need for the laborious date selection process.
We use the same CatBoost algorithms hyperparameters for all the tests.

To encode the raw time series with their corresponding algorithm, use `src/encode_tcd_EMBALGONAME.py`
Then set the path of the encoded data folder as one of the argument of `src/catboost_tcd_or_biomass.py`

### PASTIS crop classification

Pastis Crop Classification task exploits SITS rather that single date images in order to detect the crop phenology.
We use Pastis-R dataset and its variations for this DST. As some of our embedding algorithms (MMDC and pVAE) perform single-date image embeddings and do not take into account the temporal aspect, we encode Pastis SITS day-wise to produce a new encoded time series of the same temporal dimension with those embedding algorithms.

We use different DST models for predictions, depending on the input data:
- For the raw input data, as well as for MMDC and pVAE encoded series, we use U-TAE (U-net with Temporal Attention Encoder) model [1], [2] for spatio-temporal feature extraction followed by a classification layer. As in the original paper [2], we perform multi-modal predictions, as well as single-modality predictions. For multi-modal predictions, we use a late fusion feature strategy: the features are extracted for each modality separately with U-TAE model, and then concatenated and passed to the final classification layer.
- For the multi-temporal embeddings, we use a simple MLP classifier (3 dense layers with ReLU activations), as we consider that the extracted multi-temporal embeddings already contain the essential information. Note that contrary to the single-date models, we only perform single-modality embeddings.

[1] Vivien Sainte Fare Garnot, Loic Landrieu; Panoptic Segmentation of Satellite Image Time Series With Convolutional Temporal Attention Networks. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 4872-4881

[2] Vivien Sainte Fare Garnot, Loic Landrieu, Nesrine Chehata; Multi-modal temporal attention models for crop mapping from satellite time series, ISPRS Journal of Photogrammetry and Remote Sensing, Volume 187, 2022, Pages 294-305, ISSN 0924-2716,

For all the DST tasks, we train the models for 100 epochs and choose the one with the best validation metric - mIoU (mean intersection over union). We use Cross Entropy loss for model optimization.

To encode the raw time series with their corresponding algorithm, use `src/bin/encode_pastis_EMBALGONAME.py`.
Then set the output folder with the embeddings in `configs/datamodule/pastis_datamodule_encoded_ALGO_MODALITY.yaml`.
If the name of the algorithm is not indicated in filename, by default, it is MMDC.

Choose the experiment  file `config/experiment/*.yaml` that corresponds to your embeddings type and launch model training with
`python src/bin/pastis_classif_train.py experiment=experiment_name`

### BioMass and Tree Height regression

This DST exploits Sentinel-1 ASC data to regress BioMass and Tree Height parameters with MALICE algorithm.
Due to the cost of image data, we do not evaluate Sentinel-2 embeddings or/and embedding algorithms that exploit auxiliary data.

To encode the raw time series with their corresponding algorithm, use `src/bin/encode_biomass_malice.py`.
To perform the regression with an MLP model, use `src/bin/mlp_biomass.py`.

For our method, we use two separate MLP models, for AGB and FCH.
We apply weights to the high values, as they are underrepresented in our dataset.

## Code Structure

The code is organized in the following way:

├── configs\
│   ├── callbacks \
│   ├── datamodule \
│   ├── experiment \
│   ├── log_dir \
│   ├── logger \
│   ├── model \
│   ├── trainer \
│   └── train.yaml \
├── ptretrained_models\
├── src \
│   ├── bin \
│   ├── mmdc_downstream_lai \
│   ├── mmdc_downstream_pastis \
│   ├── mmdc_downstream_tcd \
│   └── mmdc_downstrteam_biomass \
└── train.py

As it can be noticed from the project structure, each DST has its associated directory with different functions
under the name `src/mmdc_downstream_*`.

Folder `bin/` contains different executable files. The name of each executable file contains the name of a DST and the embedding model.

Files that start with `encode_*` are used to produce the embeddings.

Files that end with `*_train` are used to train downstream models, such as LAI regression and Pastis crop classification.
The folder `configs/` contains different configuration files to train LAI regression and PASTIS classification deep learning models.
The subfolder `configs/experiment` contains the configuration files to launch each DST with different embeddings.
To launch a particular experiment do `python *_train.py experiment=experiment_name`

Note that for other algorithms (for ex., for embeddings computation) we directly use parser arguments in the executable files.

Biomass MLP and TCD CatBoost training are launched with files `src/bin/mlp_biomass.py` and `src/bin/catboost_tcd_or_biomass.py`.


## Requirements

You need to install the following dependencies:

- Install mmdc-singledate
- Install mt_ssl : https://src.koda.cnrs.fr/iris.dumeur/alise if you want to use ALISE/MALICE embeddings
