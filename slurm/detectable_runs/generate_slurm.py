import os
import sys
sys.path.append("/mnt/home/twagg/projects/frank-lisa/helpers/")
import const

SUBMITTER_TEMPLATE = r"""#!/bin/bash
## job name
#SBATCH --job-name=DCOTYPE_PESSLABEL_VARIATION_lisa_det_TASKID
#SBATCH --partition=cca
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --constraint=genoa
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_disBatch_0_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_disBatch_0_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

disBatch -c 2 -p /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/ Tasks_TASKID
"""

TASK_TEMPLATE = r"""#DISBATCH PREFIX source /mnt/home/twagg/.bashrc ; conda activate cogsworth ; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 ; python -u /mnt/home/twagg/projects/frank-lisa/src/sample_detectable_population.py --dco_type DCOTYPE --n_per_instance 500000 --n_instances 10 -v VARIATION -s ${DISBATCH_REPEAT_INDEX} PESSFLAG RETAIN_INTRINSIC_FLAG
#DISBATCH SUFFIX &> /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_s${DISBATCH_REPEAT_INDEX}.log
#DISBATCH REPEAT 32 start OFFSET"""

for variation in const.VARIATIONS:
    os.makedirs(os.path.join("jobs", variation), exist_ok=True)
    for dco_type in ["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"]:
        pess_list = ["", "--pessimistic"] if variation == "fiducial" else ["--pessimistic"]
        for pess_flag in pess_list:
            pess_label = "pess" if pess_flag else "opt"
            os.makedirs(os.path.join("jobs", variation, f"{dco_type}_{pess_label}"), exist_ok=True)
            for task_id in range(9):
                submitter = SUBMITTER_TEMPLATE.replace("DCOTYPE", dco_type).replace("VARIATION", variation).replace("TASKID", str(task_id)).replace("PESSLABEL", pess_label)
                task = TASK_TEMPLATE.replace("DCOTYPE", dco_type).replace("VARIATION", variation).replace("PESSFLAG", pess_flag).replace("PESSLABEL", pess_label)

                if task_id == 0:
                    task = task.replace("RETAIN_INTRINSIC_FLAG", "-r")
                    task = task.replace("32", "10").replace("OFFSET", "0")
                    submitter = submitter.replace("32", "10").replace("03:00:00", "05:00:00")
                else:
                    task = task.replace("RETAIN_INTRINSIC_FLAG", "")
                    # 32 per task, except 10 for the first
                    task = task.replace("OFFSET", str(10 + (task_id - 1) * 32))

                with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", f"job_{task_id}.slurm"), "w") as f:
                    f.write(submitter)

                with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", f"Tasks_{task_id}"), "w") as f:
                    f.write(task)

            with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", "submit.sh"), "w") as f:
                f.write("#!/bin/bash\n")
                f.write("for task_id in {0..8}; do\n")
                f.write("  sbatch job_${task_id}.slurm\n")
                f.write("  sleep 0.3\n")
                f.write("done\n")

    # create a master submit script for all DCO types
    with open(os.path.join("jobs", variation, "submit_all.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        for dco_type in ["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"]:
            pess_list = ["", "--pessimistic"] if variation == "fiducial" else ["--pessimistic"]
            for pess_flag in pess_list:
                pess_label = "pess" if pess_flag else "opt"
                f.write(f"cd {dco_type}_{pess_label}\n")
                f.write("bash submit.sh\n")
                f.write("cd ..\n")
