#!/bin/bash
## Job Name
#SBATCH --job-name=NSNS_lisa_det
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=1:30:00
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/NSNS_fiducial_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/NSNS_fiducial_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# run the distributed underworld simulation
python /mnt/home/twagg/projects/frank-lisa/src/sample_detectable_population.py \
    --folder /mnt/ceph/users/twagg/lisa-dcos/fiducial \
    --dco_type NSNS \
    --n_per_instance 500000 \
    --n_instances 20 \
    --output_folder /mnt/ceph/users/twagg/lisa-dcos/fiducial \
    -s ${SLURM_ARRAY_TASK_ID}
