#!/bin/bash
## Job Name
#SBATCH --job-name=BHNS_lisa
#SBATCH --partition=cca,gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=2:00:00
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/stroopwafel_runs/logs/BHNS_fiducial_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/stroopwafel_runs/logs/BHNS_fiducial_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cosmic

# create a log-spaced array between 1e-4 and 0.03 with 50 bins
METS=($(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-4, np.log10(0.03), 50).round(5))))"))

# select a metallicity based on the SLURM_ARRAY_TASK_ID
MET=${METS[$SLURM_ARRAY_TASK_ID]}

echo "Starting BHNS simulation with metallicity: $MET"

# run the distributed underworld simulation
python /mnt/home/twagg/projects/frank-lisa/src/create_dco_population.py \
    --metallicity $MET \
    --inifile /mnt/home/twagg/projects/frank-lisa/src/params.ini \
    --total_systems 2000000 \
    --batch_size 25000 \
    --dco_type BHNS \
    --nproc 64 \
    --output_path /mnt/ceph/users/twagg/lisa-dcos/fiducial/BHNS_Z_$MET.h5
