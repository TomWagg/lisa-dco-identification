#!/bin/bash
## job name
#SBATCH --job-name=BHBH_lisa_det
#SBATCH --partition=cca,gen
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/disBatch_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/disBatch_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

# run all 100 tasks through disBatch, drawing from the shared core pool
disBatch -p /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/ Tasks