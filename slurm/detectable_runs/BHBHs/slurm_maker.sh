for i in 0 1 2 3 4 5 6 7; do
    sed -e "s/_det_0/_det_${i}/" -e "s/Tasks_0/Tasks_${i}/" -e "s/disBatch_0_/disBatch_${i}_/" _submitter_template.slurm > job_${i}.slurm
done