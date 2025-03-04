#!/bin/bash
#SBATCH --job-name=prepare_tile
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -N 1                       # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=8                       # number of tasks ( or --tesks=8)
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --partition=gpu_a100       # partition
#SBATCH --qos=gpu_all              # QoS
#SBATCH --time=2:00:00            # Walltime
#SBATCH --mem-per-cpu=12G          # memory per core
#SBATCH --account=cesbio           # MANDATORY : account (launch myaccounts to list your accounts)
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc

# be sure no modules loaded
module purge

export SCRATCH=/work/scratch/data/${USER}
export SRCDIR=${SCRATCH}/src/MMDC/mmdc-downstream
export MMDC_INIT=${SRCDIR}/mmdc_init.sh
export WORKING_DIR=${SCRATCH}/MMDC/jobs

cd ${WORKING_DIR}

module purge
module load conda
conda activate alise

HYDRA_FULL_ERROR=1 python ${SRCDIR}/src/bin/encode_pastis_alise.py >> output_$SLURM_JOBID.log
