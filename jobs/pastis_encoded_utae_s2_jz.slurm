#!/bin/bash
#SBATCH --job-name=pastis_encoded_s2_utae
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -A ttd@a100
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# be sure no modules loaded
module purge

export SRCDIR=$WORK/src/MMDC/mmdc-downstream
export MMDC_INIT=${SRCDIR}/mmdc_init_jz.sh
export WORKING_DIR=$WORK/results/MMDC/jobs

cd ${WORKING_DIR}
source ${MMDC_INIT}

HYDRA_FULL_ERROR=1 python   ${SRCDIR}/train.py experiment=pastis_encoded_utae_s2 >> output_$SLURM_JOBID.log
