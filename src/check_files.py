
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
            get_n_target(
                folder=folder, dco_type=dco_type, f_trunc=f_truncs[dco_type],
                MW_weights=MW_weights, MW_SF=MW_SF)

    return


def get_n_target(folder, dco_type, f_trunc, MW_weights, MW_SF):
    print(f"Calculating n_target for {dco_type} in {folder}")

    # loop over metallicities
    for Z in const.Z_BIN_CENTRES:
        if not os.path.isfile(f"{folder}/stroopwafel_files/{dco_type}_Z_{Z}.h5"):
            print(f"    File {folder}/stroopwafel_files/{dco_type}_Z_{Z}.h5 does not exist, skipping")
            continue


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


if __name__ == "__main__":
    main()

# python check_files.py -b /mnt/ceph/users/twagg/lisa-dcos -f fiducial alpha_low alpha_high maltsev no_ecsn qcflag2 -d BHBH BHNS BHWD NSNS NSWD