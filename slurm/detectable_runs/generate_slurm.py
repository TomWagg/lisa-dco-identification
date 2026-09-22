import os
import sys
sys.path.append("/mnt/home/twagg/projects/frank-lisa/helpers/")
import const

SUBMITTER_TEMPLATE = r"""#!/bin/bash
## job name
#SBATCH --job-name=DCOTYPE_PESSLABEL_VARIATION_lisa_det
#SBATCH --partition=cca
#SBATCH --ntasks=266
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --constraint=genoa
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_disBatch_0_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_disBatch_0_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

disBatch -c 2 -p /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/ Tasks
"""

TASK_TEMPLATE = r"""#DISBATCH PREFIX source /mnt/home/twagg/.bashrc ; conda activate cogsworth ; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 ; python -u /mnt/home/twagg/projects/frank-lisa/src/sample_detectable_population.py --dco_type DCOTYPE --n_per_instance 500000 --n_instances 10 -v VARIATION -s ${DISBATCH_REPEAT_INDEX} PESSFLAG RETAIN_INTRINSIC_FLAG
#DISBATCH SUFFIX &> /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCOTYPE_PESSLABEL_VARIATION_s${DISBATCH_REPEAT_INDEX}.log
#DISBATCH REPEAT 256 start OFFSET"""

for variation in const.VARIATIONS:
    os.makedirs(os.path.join("jobs", variation), exist_ok=True)
    for dco_type in ["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"]:
        pess_list = ["", "--pessimistic"] if variation == "fiducial" else ["--pessimistic"]
        for pess_flag in pess_list:
            pess_label = "pess" if pess_flag else "opt"
            os.makedirs(os.path.join("jobs", variation, f"{dco_type}_{pess_label}"), exist_ok=True)

            submitter = SUBMITTER_TEMPLATE.replace("DCOTYPE", dco_type).replace("VARIATION", variation).replace("PESSLABEL", pess_label)
            if variation != "fiducial":
                submitter = submitter.replace("266", "256")
            if variation == "alpha_high":
                submitter = submitter.replace("#SBATCH --constraint=genoa", "#SBATCH --constraint=genoa\n#SBATCH --ntasks-per-node=32")
            with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", f"job.slurm"), "w") as f:
                f.write(submitter)

            if os.path.exists(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", "Tasks")):
                os.remove(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", "Tasks"))

            # only the fiducial model needs to handle the first 10 instances with the -r flag
            task_id_list = [0, 1] if variation == "fiducial" else [1]

            for task_id in task_id_list:
                task = TASK_TEMPLATE.replace("DCOTYPE", dco_type).replace("VARIATION", variation).replace("PESSFLAG", pess_flag).replace("PESSLABEL", pess_label)

                if task_id == 0:
                    task = task.replace("RETAIN_INTRINSIC_FLAG", "-r")
                    task = task.replace("256", "10").replace("OFFSET", "0")
                else:
                    task = task.replace("RETAIN_INTRINSIC_FLAG", "")
                    # 32 per task, except 10 for the first
                    task = task.replace("OFFSET", "10" if variation == "fiducial" else "0")

                with open(os.path.join("jobs", variation, f"{dco_type}_{pess_label}", f"Tasks"), "a") as f:
                    f.write(task + "\n")

    # create a master submit script for all DCO types
    with open(os.path.join("jobs", variation, "submit_all.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        for dco_type in ["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"]:
            pess_list = ["", "--pessimistic"] if variation == "fiducial" else ["--pessimistic"]
            for pess_flag in pess_list:
                pess_label = "pess" if pess_flag else "opt"
                f.write(f"cd {dco_type}_{pess_label}\n")
                f.write(f"sbatch job.slurm\n")
                f.write("cd ..\n")
