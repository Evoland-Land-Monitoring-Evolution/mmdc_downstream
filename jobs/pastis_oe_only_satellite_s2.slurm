#!/bin/bash
#SBATCH --job-name=pastis_utae
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -A ttd@a100
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:00             # Walltime 24h

# be sure no modules loaded
module purge

export SCRATCH=/work/scratch/data/${USER}
export SRCDIR=${SCRATCH}/src/MMDC/mmdc-downstream
export MMDC_INIT=${SRCDIR}/mmdc_init.sh
export WORKING_DIR=${SCRATCH}/MMDC/jobs

cd ${WORKING_DIR}
source ${MMDC_INIT}

HYDRA_FULL_ERROR=1 python   ${SRCDIR}/train.py experiment=pastis_oe_only_satellite_s2 >> output_$SLURM_JOBID.log
