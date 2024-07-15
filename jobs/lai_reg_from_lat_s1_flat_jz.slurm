#!/bin/bash
#SBATCH --job-name=lai_reg_s2_flat
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -A ttd@a100
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00             # Walltime 24h
#SBATCH --qos=qos_gpu-dev

# be sure no modules loaded
module purge

export SRCDIR=$WORK/src/MMDC/mmdc-downstream
export MMDC_INIT=${SRCDIR}/mmdc_init_jz.sh
export WORKING_DIR=$WORK/results/MMDC/jobs


cd ${WORKING_DIR}
source ${MMDC_INIT}

HYDRA_FULL_ERROR=1 python   ${SRCDIR}/train.py experiment=lai_reg_latent_s1_flat >> output_$SLURM_JOBID.log
