
import h5py as h5
import os
import pandas as pd
import numpy as np

import gala.potential as gp
import cogsworth

import sys
sys.path.append("../helpers")

from truncation import get_f_trunc
import const

import argparse


def get_normalisations(folders, dco_types, f_trunc_kwargs={}, MW_SF=10.4e10):
    if isinstance(folders, str):
        folders = [folders]
    if isinstance(dco_types, str):
        dco_types = [dco_types]

    # First calculate things that are constant across all models
    # ----------------------------------------------------------

    # MW metallicity weights
    MW_weights_path = "/mnt/home/twagg/projects/frank-lisa/data/MW_Z_weights.npy"
    if os.path.isfile(MW_weights_path):
        MW_weights = np.load(MW_weights_path)
    else:
        g = cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential2022())
        g.sample(10_000_000)
        counts, _ = np.histogram(g.Z.value, bins=const.Z_BIN_EDGES)
        MW_weights = counts / np.sum(counts)
        np.save(MW_weights_path, MW_weights)

    # truncation factors
    f_truncs = {dco_type: get_f_trunc(m1_low=const.M1_MIN[dco_type], **f_trunc_kwargs)
                for dco_type in dco_types}

    n_targets = {dco_type: [] for dco_type in dco_types}
    pessimistic_n_targets = {f"{dco_type}_pessimistic": [] for dco_type in dco_types}
    for folder in folders:
        for dco_type in dco_types:
            n_target, pessimistic_n_target = get_n_target(
                folder=folder, dco_type=dco_type, f_trunc=f_truncs[dco_type],
                MW_weights=MW_weights, MW_SF=MW_SF)
            n_targets[dco_type].append(n_target)
            pessimistic_n_targets[f"{dco_type}_pessimistic"].append(pessimistic_n_target)

    both_n_targets = {**n_targets, **pessimistic_n_targets}

    # get model names from the last folder in the path, account for trailing slashes
    model_names = [os.path.basename(os.path.normpath(folder)) for folder in folders]

    df = pd.DataFrame(both_n_targets, index=model_names)
    return df


def get_n_target(folder, dco_type, f_trunc, MW_weights, MW_SF):
    print(f"\n\nCalculating n_target for {dco_type} in {folder}")
    # read in the formation rows for this folder and dco_type
    all_formation_rows = pd.read_hdf(f"{folder}/{dco_type}_formation_rows.h5", key="formation_rows")

    # track the star forming mass and n_target at each metallicity bin
    star_forming_mass_at_Z = []
    n_target_at_Z = []
    pessimistic_n_target_at_Z = []

    # loop over metallicities
    for Z in const.Z_BIN_CENTRES:
        print(f"  Processing Z = {Z}")
        # load the file and calculate the total star forming mass, accounting for weights
        with h5.File(f"{folder}/stroopwafel_files/{dco_type}_Z_{Z}.h5", "r") as f:
            mass_1, q = f["stroopwafel"]["samples"][...][:, 0], f["stroopwafel"]["samples"][...][:, 1]
            mass_2 = mass_1 * q
            weights = f["stroopwafel"]["weights"][...]
            star_forming_mass_at_Z.append(np.sum((mass_1 + mass_2) * weights))

        # track the number of target systems, again accounting for weights
        formation_rows_Z = all_formation_rows[all_formation_rows["metallicity"] == Z]
        pessimistic_formation_rows_Z = formation_rows_Z[~formation_rows_Z["remove_if_pessimistic"]]
        n_target_at_Z.append(np.sum(formation_rows_Z["weights"].values))
        pessimistic_n_target_at_Z.append(np.sum(pessimistic_formation_rows_Z["weights"].values))

    star_forming_mass_at_Z = np.array(star_forming_mass_at_Z)
    n_target_at_Z = np.array(n_target_at_Z)
    pessimistic_n_target_at_Z = np.array(pessimistic_n_target_at_Z)

    n_target = np.sum(n_target_at_Z * MW_weights / (star_forming_mass_at_Z / f_trunc)) * MW_SF
    pessimistic_n_target = np.sum(pessimistic_n_target_at_Z * MW_weights / (star_forming_mass_at_Z / f_trunc)) * MW_SF
    return n_target, pessimistic_n_target

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_folder", type=str, required=True, help="Base folder containing the DCO folders")
    parser.add_argument("-f", "--folders", type=str, nargs="+", help="Folders to process")
    parser.add_argument("-d", "--dco_types", type=str, nargs="+", help="DCO types to process")

    args = parser.parse_args()

    if args.folders is None:
        raise ValueError("Please provide a list of folders to process using the -f or --folders argument.")
    if args.dco_types is None:
        raise ValueError("Please provide a list of DCO types to process using the -d or --dco_types argument.")

    folders = [os.path.join(args.base_folder, folder) for folder in args.folders]
    df = get_normalisations(folders, args.dco_types)
    print(df)

    df.to_hdf(f"{args.base_folder}/normalisations.h5", key="n_targets", mode="w")


if __name__ == "__main__":
    main()

# python norm.py -b /mnt/ceph/users/twagg/lisa-dcos -f fiducial -d BHBH