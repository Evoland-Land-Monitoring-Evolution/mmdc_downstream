#!/bin/bash
#SBATCH --job-name=tcd
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -N 1                       # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=8                       # number of tasks ( or --tesks=8)
#SBATCH --time=12:00:00            # Walltime
#SBATCH --mem-per-cpu=92G          # memory per core
#SBATCH --account=cesbio           # MANDATORY : account (launch myaccounts to list your accounts)
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc



# be sure no modules loaded
module purge

export SCRATCH=/work/scratch/data/${USER}
export SRCDIR=${SCRATCH}/src/MMDC/mmdc-downstream
export MMDC_INIT=${SRCDIR}/mmdc_init.sh
export WORKING_DIR=${SCRATCH}/MMDC/jobs

cd ${WORKING_DIR}
source ${MMDC_INIT}

HYDRA_FULL_ERROR=1 python   ${SRCDIR}/src/mmdc_downstream_tcd/tile_processing/compute_tcd_s2.py >> output_$SLURM_JOBID.log
