for i in 0 1 2 3 4 5 6 7; do
    sbatch job_${i}.slurm
    sleep 1
done