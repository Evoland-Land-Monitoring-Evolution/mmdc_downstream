## Overview

This code is designed to validate the quality of image embeddings on different downstream tasks (DST).

We test the images produced by the following embedding algorithms:

- MMDC
- ALISE
- MALICE
- MALICE Aux
- pVAE

The following DST tasks were evaluated:

- Leaf-area-Index (LAI) regression
- Tree Cover Density (TCD) regression
- PASTIS crop classification
- BioMass and Tree Height regression

## What's inside?

This repository contains multiple scripts to:

- compute and save the embeddings with different algorithms,
- perform DST with those embeddings.

Note that we do not compute the embeddings "on the go" while training DST algorithms, but always pre-compute that embeddings.

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
├── src \
│   ├── bin \
│   ├── mmdc_downstream_lai \
│   ├── mmdc_downstream_pastis \
│   ├── mmdc_downstream_tcd \
│   └── mmdc_downstrteam_biomass \
└── train.py

As it can be noticed from the project structure, each DST has its associated directory with different functions
under the name `src/mmdc_downstream_*`.

Folder `bin/` contains different executable files. The name of each executable file contains the name of DST.
Files that start with `encode_*` are used to produce the embeddings.

The folder `configs/` contains different configuration files to train LAI regression and PASTIS classification deep learning models.
The subfolder `configs/experiment` contains the configuration files to launch each DST with different embeddings.

Note that for other algorithms (for ex., for embeddings computation) we directly use parser arguments
