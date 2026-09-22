import os
import pandas as pd
import numpy as np
import cogsworth
import argparse
from time import time

def get_pop_and_f_detect(folder, dco_type):
    f_detects = []
    pops = []

    pop_files = []
    f_det_files = []

    for file in os.listdir(folder):
        if "test" in file:
            continue
        if file.startswith(f"{dco_type}_in_band"):
            pop_files.append(file)
        elif file.startswith(f"{dco_type}_f_detect") and file.endswith(".h5"):
            f_det_files.append(file)

    assert len(pop_files) == len(f_det_files), f"Number of population files ({len(pop_files)}) does not match number of f_detect files ({len(f_det_files)})"

    pop_files = np.sort(pop_files)
    f_det_files = np.sort(f_det_files)

    big_file_total_so_far = 0

    for i, (pop_file, f_det_file) in enumerate(zip(pop_files, f_det_files)):
        is_big_file = int(pop_file.split("_")[-1].split(".")[0]) < 10
        if is_big_file:
            if big_file_total_so_far > 1e6:
                print(f"  Skipping {pop_file} and {f_det_file} because big_file_total_so_far is {big_file_total_so_far}")
                continue
        pops.append(cogsworth.pop.load(os.path.join(folder, pop_file), parts=["initial_binaries", "initial_galaxy", "stellar_evolution", "galactic_orbits"]))
        f_detects.append(pd.read_hdf(os.path.join(folder, f_det_file)))
        print(f"  Loaded {pop_file} and {f_det_file}, {len(pops[-1])} binaries")
        if is_big_file:
            big_file_total_so_far += len(pops[-1])

    running_instance_total = 0
    for pop in pops:
        pop.orbits
        pop.final_pos, pop.final_vel
        max_inst = pop.bpp["MW_instance"].max()
        pop.bpp["MW_instance"] += running_instance_total
        running_instance_total += max_inst + 1
    pop = cogsworth.pop.concat(*pops)

    pop.initial_galaxy
    sfh = cogsworth.sfh.StarFormationHistory()
    for var in ["_x", "_y", "_z", "_v_x", "_v_y", "_v_z", "_tau", "_Z"]:
        setattr(sfh, var, getattr(pop.initial_galaxy, var))
    pop._initial_galaxy = sfh

    f_detect = pd.concat(f_detects, ignore_index=True)

    return pop, f_detect


def main():
    parser = argparse.ArgumentParser(description="Concatenate detectable populations and f_detect arrays.")
    parser.add_argument("-f", "--folder", type=str, required=True, help="Folder containing the detectable populations and f_detect arrays.")
    parser.add_argument("-d", "--dco_type", type=str, nargs="+", required=True, help="Type of DCO (e.g., NSWD, NSNS, BHWD, BHNS, BHBH).")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output path for the concatenated population and f_detect array.")
    parser.add_argument("-O", "--overwrite", action="store_true", help="Overwrite existing files if they exist.")
    parser.add_argument('-t', '--trim-pessimistic', action='store_true', help="Trim the pessimistic flag from the file name")

    args = parser.parse_args()

    print(f"Concatenating detectable populations and f_detect arrays for DCO types: {args.dco_type}")

    for dco_type in args.dco_type:
        print(f"Processing {dco_type}...")
        output_dco_type = dco_type
        if args.trim_pessimistic:
            output_dco_type = dco_type.replace("_pessimistic", "")
        output_pop_path = os.path.join(args.output_path, f"{output_dco_type}_in_band.h5")
        output_f_detect_path = os.path.join(args.output_path, f"{output_dco_type}_f_detect.h5")
        if os.path.exists(output_pop_path) and not args.overwrite:
            raise FileExistsError(f"{output_pop_path} already exists. Use --overwrite to overwrite.")
        if os.path.exists(output_f_detect_path) and not args.overwrite:
            raise FileExistsError(f"{output_f_detect_path} already exists. Use --overwrite to overwrite.")

        start = time()

        pop, f_detect = get_pop_and_f_detect(args.folder, dco_type)

        print(f"  Concatenation took {time() - start:.2f} seconds.")
        start = time()

        pop.save(output_pop_path, overwrite=True)
        f_detect.to_hdf(output_f_detect_path, key="f_detect", mode="w")
        print(f"  Saving took {time() - start:.2f} seconds.")


if __name__ == "__main__":
    main()


# python concat_detectable_pops.py -f "/mnt/ceph/users/twagg/lisa-dcos/fiducial/detection_files" -d "NSWD BHWD" -o "/mnt/ceph/users/twagg/lisa-dcos/fiducial/" -O