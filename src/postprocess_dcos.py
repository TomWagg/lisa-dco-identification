from cosmic.output import load_initC, save_initC
import h5py as h5
import os
import pandas as pd
import numpy as np
from time import time
import argparse

kstar_masks = {
    "NSWD": ([13], [10, 11, 12]),
    "NSNS": ([13], [13]),
    "BHWD": ([14], [10, 11, 12]),
    "BHNS": ([14], [13]),
    "BHBH": ([14], [14])
}


def postprocess(dco_type, folder):
    input_folder = os.path.join(folder, "stroopwafel_files")

    start_time = time()
    print(f"Processing {dco_type}...")
    kstars = kstar_masks[dco_type]

    files = []
    for file in os.listdir(input_folder):
        if file.startswith(f"{dco_type}_Z_"):
            files.append(file)

    formation_rows_list = []
    initC_list = []
    initC_at_DCO_list = []

    files = np.sort(files)
    for file in files:
        file_start_time = time()
        print(f"\n\nStarting {file}")

        # load initC and bpp
        initC = load_initC(f"{input_folder}/{file}", key="initC")
        bpp = pd.read_hdf(f"{input_folder}/{file}", key="bpp")

        print(f"    [{time() - file_start_time:.2f}s] Loaded initC and bpp")
        lap = time()

        # find the DCO formation rows
        k_lo, k_hi = kstars
        mask = ((
            (bpp["kstar_1"].isin(k_lo)) & (bpp["kstar_2"].isin(k_hi))
        ) | (
            (bpp["kstar_1"].isin(k_hi)) & (bpp["kstar_2"].isin(k_lo))
        )) & (bpp["sep"] > 0)

        # if there are none then skip
        if mask.sum() == 0:
            print(f"  No {dco_type} systems found in this file.")
        else:
            print(f"    [{time() - lap:.2f}s] Found {mask.sum()} {dco_type} systems in this file.")
            lap = time()

            # otherwise record the formation rows and their metallicities
            formation_rows_all_cols = bpp[mask].drop_duplicates(subset="bin_num")
            formation_rows = formation_rows_all_cols[["tphys", "mass_1", "mass_2", "kstar_1", "kstar_2", "sep", "ecc", "bin_num"]]
            formation_rows["metallicity"] = initC["metallicity"].iloc[0]

            # read in the AIS weights from STROOPWAFEL
            with h5.File(f"{input_folder}/{file}", "r") as f:
                formation_rows["weights"] = f["stroopwafel"]["weights"][...][f["stroopwafel"]["is_hit"][...]]

            initC_list.append(initC.loc[formation_rows_all_cols["bin_num"]])

            print(f"    [{time() - lap:.2f}s] Found formation rows and weights")
            lap = time()

            # find out whether each binary would have been masked in the pessimistic CE
            dco_forming_bpp = bpp.loc[formation_rows["bin_num"]]

            pessimistic = (dco_forming_bpp["evol_type"] == 7) & (
                ((dco_forming_bpp["RRLO_1"] > 1) & (dco_forming_bpp["kstar_1"].isin([0,1,2,7,8,10,11,12]))) |
                ((dco_forming_bpp["RRLO_2"] > 1) & (dco_forming_bpp["kstar_2"].isin([0,1,2,7,8,10,11,12])))
            )
            fail_pessimistic = dco_forming_bpp[pessimistic]["bin_num"]
            formation_rows["remove_if_pessimistic"] = formation_rows["bin_num"].isin(fail_pessimistic)

            # if the DCO contains a WD, record the time at which RLOF occurs if it does (NaN otherwise)
            if dco_type in ["NSWD", "BHWD"]:
                formation_rows["rlof_time"] = np.nan
                formation_rows["rlof_sep"] = np.nan
                rlof_stats = dco_forming_bpp[["tphys", "sep"]][((
                    (dco_forming_bpp["kstar_1"].isin(k_lo)) & (dco_forming_bpp["kstar_2"].isin(k_hi))
                ) | (
                    (dco_forming_bpp["kstar_1"].isin(k_hi)) & (dco_forming_bpp["kstar_2"].isin(k_lo))
                )) & (dco_forming_bpp["sep"] > 0) & (dco_forming_bpp["evol_type"] == 3)]
                formation_rows.loc[rlof_stats.index, "rlof_time"] = rlof_stats["tphys"]
                formation_rows.loc[rlof_stats.index, "rlof_sep"] = rlof_stats["sep"]

                # record the initC at DCO formation for each binary
                initC_at_formation = initC.loc[formation_rows_all_cols["bin_num"]]
                shared_columns = initC.columns.intersection(formation_rows_all_cols.columns)
                initC_at_formation[shared_columns] = formation_rows_all_cols[shared_columns]
                initC_at_DCO_list.append(initC_at_formation)

            formation_rows_list.append(formation_rows)

        print(f"  [{time() - file_start_time:.2f}s] Finishing processing {file}")

    all_formation_rows = pd.concat(formation_rows_list, ignore_index=True)
    all_formation_rows.to_hdf(f"{folder}/{dco_type}_at_formation.h5", key="formation_rows")

    all_initC = pd.concat(initC_list, ignore_index=True)
    save_initC(f"{folder}/{dco_type}_at_formation.h5", all_initC, key="initC")

    if len(initC_at_DCO_list) > 0:
        all_initC_at_DCO = pd.concat(initC_at_DCO_list, ignore_index=True)
        save_initC(f"{folder}/{dco_type}_at_formation.h5", all_initC_at_DCO,
                key="initC_at_DCO", settings_key="initC_at_DCO_settings")

    kick_infos = [pd.read_hdf(f"{input_folder}/{file}", key="kick_info")[
        ["tphys", "star", "delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1", "bin_num"]
    ] for file in os.listdir(input_folder) if file.startswith(f"{dco_type}_Z_")]
    all_kick_infos = pd.concat(kick_infos, ignore_index=True)
    all_kick_infos.index = all_kick_infos.index.values // 2
    all_kick_infos.to_hdf(f"{folder}/{dco_type}_at_formation.h5", key="kick_info")

    print(f"Finished processing {dco_type} in {time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Postprocess DCOS")
    parser.add_argument("-d", "--dco_type", type=str, choices=kstar_masks.keys(), help="Type of DCO to process")
    parser.add_argument("-f", "--folder", type=str, required=True, help="Folder containing the STROOPWAFEL files")
    args = parser.parse_args()

    postprocess(args.dco_type, args.folder)


if __name__ == "__main__":
    main()

# python postprocess_dcos.py -d BHBH -f /mnt/ceph/users/twagg/lisa-dcos/fiducial