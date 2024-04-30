#!/bin/bash
#SBATCH --job-name=pastis_oe_utae_concat_s1_asc_desc
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -A ttd@a100
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00             # Walltime 24h


# be sure no modules loaded
module purge

export SRCDIR=$WORK/src/MMDC/mmdc-downstream
export MMDC_INIT=${SRCDIR}/mmdc_init_jz.sh
export WORKING_DIR=$WORK/results/MMDC/jobs

cd ${WORKING_DIR}
source ${MMDC_INIT}

HYDRA_FULL_ERROR=1 python   ${SRCDIR}/train.py experiment=pastis_oe_utae_concat_s1_asc_desc.slurm >> output_$SLURM_JOBID.log
