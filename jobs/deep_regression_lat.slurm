#!/bin/bash
#SBATCH --job-name=lai_deep
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -A ttd@a100
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=5:00:00             # Walltime 24h


# be sure no modules loaded
module purge

export SRCDIR=$WORK/src/MMDC/mmdc-downstream
export MMDC_INIT=${SRCDIR}/mmdc_init.sh
export WORKING_DIR=$WORK/results/LAI/jobs

cd ${WORKING_DIR}
source ${MMDC_INIT}

HYDRA_FULL_ERROR=1 srun python ${SRCDIR}/src/mmdc_downstream/models/test_baselines/deep_regression_lat_new.py
 >> output_$SLURM_JOBID.log
