import os

SUBMITTER_TEMPLATE = r"""#!/bin/bash
## job name
#SBATCH --job-name=DCOTYPE_PESSLABEL_VARIATION_lisa_det_TASKID
#SBATCH --partition=cca
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_disBatch_0_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_disBatch_0_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

disBatch -c 2 -p /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/ Tasks_TASKID
"""

TASK_TEMPLATE = r"""#DISBATCH PREFIX source /mnt/home/twagg/.bashrc ; conda activate cogsworth ; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 ; python -u /mnt/home/twagg/projects/frank-lisa/src/sample_detectable_population.py --folder /mnt/ceph/users/twagg/lisa-dcos/VARIATION --dco_type DCOTYPE --n_per_instance 500000 --n_instances 10 --output_folder /mnt/ceph/users/twagg/lisa-dcos/VARIATION -s ${DISBATCH_REPEAT_INDEX} PESSFLAG
#DISBATCH SUFFIX &> /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_s${DISBATCH_REPEAT_INDEX}.log
#DISBATCH REPEAT 32 start OFFSET"""

for variation in ["fiducial"]:
    os.makedirs(os.path.join("jobs", variation), exist_ok=True)
    for dco_type in ["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"]:
        for pess_flag in ["", "--pessimistic"]:
            pess_label = "pess" if pess_flag else "opt"
            os.makedirs(os.path.join("jobs", variation, f"{dco_type}_{pess_label}"), exist_ok=True)
            for task_id in range(8):
                submitter = SUBMITTER_TEMPLATE.replace("DCOTYPE", dco_type).replace("VARIATION", variation).replace("TASKID", str(task_id)).replace("PESSLABEL", pess_label)
                task = TASK_TEMPLATE.replace("DCOTYPE", dco_type).replace("VARIATION", variation).replace("PESSFLAG", pess_flag).replace("PESSLABEL", pess_label).replace("OFFSET", str(task_id * 32))

                with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", f"job_{task_id}.slurm"), "w") as f:
                    f.write(submitter)

                with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", f"Tasks_{task_id}"), "w") as f:
                    f.write(task)

            with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", "submit.sh"), "w") as f:
                f.write("#!/bin/bash\n")
                f.write("for task_id in {0..7}; do\n")
                f.write("  sbatch job_${task_id}.slurm\n")
                f.write("  sleep 0.3\n")
                f.write("done\n")
