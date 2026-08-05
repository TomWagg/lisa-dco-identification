#!/bin/bash
## job name
#SBATCH --job-name=BHBH_lisa_det_0
#SBATCH --partition=cca,gen
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/BHBH_disBatch_0_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/BHBH_disBatch_0_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

disBatch -c 2 -p /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/ Tasks_0