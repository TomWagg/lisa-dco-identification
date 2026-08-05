TEMPLATE = r"""#!/bin/bash
## Job Name
#SBATCH --job-name=DCO_TYPE_lisa_det
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=3:00:00
#SBATCH -o /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCO_TYPE_fiducial_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/frank-lisa/slurm/detectable_runs/logs/DCO_TYPE_fiducial_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

# run the distributed underworld simulation
python /mnt/home/twagg/projects/frank-lisa/src/sample_detectable_population.py \
    --folder /mnt/ceph/users/twagg/lisa-dcos/fiducial \
    --dco_type DCO_TYPE \
    --n_per_instance 500000 \
    --n_instances 20 \
    --output_folder /mnt/ceph/users/twagg/lisa-dcos/fiducial \
    -s ${SLURM_ARRAY_TASK_ID}
"""

for dco_type in ["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"]:
    with open(f"f_detect_{dco_type}s.slurm", "w") as f:
        f.write(TEMPLATE.replace("DCO_TYPE", dco_type))
