import os
import numpy as np
import cogsworth
import argparse


def get_pop_and_f_detect(folder, dco_type):
    f_detects = []
    pops = []

    pop_files = []
    f_det_files = []

    for file in os.listdir(folder):
        if file.startswith(f"{dco_type}_detectable"):
            pop_files.append(file)
        elif file.startswith(f"{dco_type}_f_detect") and file.endswith(".npy"):
            f_det_files.append(file)

    pop_files = np.sort(pop_files)
    f_det_files = np.sort(f_det_files)

    for i, (pop_file, f_det_file) in enumerate(zip(pop_files, f_det_files)):
        print(f"Loading {pop_file} and {f_det_file}")
        pops.append(cogsworth.pop.load(os.path.join(folder, pop_file)))
        f_detects.append(np.load(os.path.join(folder, f_det_file)))

    for pop in pops:
        pop.final_pos, pop.final_vel
    pop = cogsworth.pop.concat(*pops)

    pop.initial_galaxy
    sfh = cogsworth.sfh.StarFormationHistory()
    for var in ["_x", "_y", "_z", "_v_x", "_v_y", "_v_z", "_tau", "_Z"]:
        setattr(sfh, var, getattr(pop.initial_galaxy, var))
    pop._initial_galaxy = sfh

    f_detect = np.concatenate(f_detects)

    return pop, f_detect


def main():
    parser = argparse.ArgumentParser(description="Concatenate detectable populations and f_detect arrays.")
    parser.add_argument("-f", "--folder", type=str, required=True, help="Folder containing the detectable populations and f_detect arrays.")
    parser.add_argument("-d", "--dco_type", type=str, required=True, help="Type of DCO (e.g., NSWD, NSNS, BHWD, BHNS, BHBH).")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output path for the concatenated population and f_detect array.")
    parser.add_argument("-O", "--overwrite", action="store_true", help="Overwrite existing files if they exist.")

    args = parser.parse_args()

    output_pop_path = os.path.join(args.output_path, f"{args.dco_type}_detectable.h5")
    output_f_detect_path = os.path.join(args.output_path, f"{args.dco_type}_f_detect.npy")
    if os.path.exists(output_pop_path) and not args.overwrite:
        raise FileExistsError(f"{output_pop_path} already exists. Use --overwrite to overwrite.")
    if os.path.exists(output_f_detect_path) and not args.overwrite:
        raise FileExistsError(f"{output_f_detect_path} already exists. Use --overwrite to overwrite.")

    pop, f_detect = get_pop_and_f_detect(args.folder, args.dco_type)

    pop.save(output_pop_path)
    np.save(output_f_detect_path, f_detect)


if __name__ == "__main__":
    main()


# python concat_detectable_pops.py -f "/mnt/ceph/users/twagg/lisa-dcos/fiducial/detection_files" -d "BHBH" -o "/mnt/ceph/users/twagg/lisa-dcos/fiducial/"