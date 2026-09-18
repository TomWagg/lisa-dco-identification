import os.path
import argparse
import time

import pandas as pd
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
import h5py as h5

import sys
sys.path.append("/mnt/home/twagg/projects/frank-lisa/helpers/")
import const
from sfh import SB15_PhiCut

import gala.potential as gp
import cogsworth
from cosmic.evolve import Evolve
from cosmic.output import load_initC


# CONSTANTS STUFF

SB15 = cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential(version='v2'))

WEIGHTS_PATH = "/mnt/home/twagg/projects/frank-lisa/data/MW_Z_weights.npy"
if os.path.isfile(WEIGHTS_PATH):
    MW_weights = np.load(WEIGHTS_PATH)
else:
    SB15.sample(10_000_000)
    counts, _ = np.histogram(SB15.Z.value, bins=const.Z_BIN_EDGES)
    MW_weights = counts / np.sum(counts)
    np.save(WEIGHTS_PATH, MW_weights)



def evolve_milky_way_instance(all_formation_rows, all_kick_infos, all_initCs, all_initC_at_DCOs=None,
                              n_per_instance=500_000, oversample_factor=5,
                              retain_intrinsic=False, variation="fiducial"):
    lap = time.time()

    # draw a random sample, weighted by the Milky Way metallicity distribution
    rand_sample = all_formation_rows.sample(n_per_instance, weights="MW_Z_weight", replace=True)
    rand_kicks = all_kick_infos.loc[rand_sample.index]
    rand_initC = all_initCs.loc[rand_sample.index]
    rand_initC_at_DCO = all_initC_at_DCOs.loc[rand_sample.index] if all_initC_at_DCOs is not None else None

    print(f"  [{time.time() - lap:03.1f}s] Sampled {n_per_instance} systems from the formation rows")
    lap = time.time()

    # work out the target number of systems in each metallicity bin, for galaxy sampling
    target_counts, _ = np.histogram(rand_sample["metallicity"].values, bins=const.Z_BIN_EDGES)
    sample_size = target_counts.sum() * oversample_factor

    # loop variables
    counts, rnd, sfhs = np.zeros_like(target_counts), 0, []

    print(f"  [{time.time() - lap:03.1f}s] Starting to sample from the SFH to get {target_counts.sum()} systems in the right metallicity bins...")
    lap = time.time()

    # continue until we have enough systems in each metallicity bin
    while np.sum(counts) < np.sum(target_counts):
        # find the maximum metallicity that doesn't have a full bin
        max_Z = const.Z_BIN_CENTRES[counts < target_counts].max()
    
        # max Z is below 1e-3 then switch to the phi cut version
        if max_Z <= 5e-4:
            print(f"    Using phi cut of 3.0 for max_Z <= 5e-4, with max_Z = {max_Z}")
            SB15 = SB15_PhiCut(phi_cut=3.0, potential=gp.MilkyWayPotential(version="v2"))
        elif max_Z <= 1e-3:
            print(f"    Using phi cut of 2.0 for max_Z <= 1e-3, with max_Z = {max_Z}")
            SB15 = SB15_PhiCut(phi_cut=2.0, potential=gp.MilkyWayPotential(version="v2"))
        else:
            print(f"    Using phi cut of 0.0 for max_Z > 1e-3, with max_Z = {max_Z}")
            SB15 = SB15_PhiCut(phi_cut=0.0, potential=gp.MilkyWayPotential(version="v2"))
        # sample from the SFH, clipping metallicities to the range of the COSMIC sim
        SB15.sample(int(sample_size))
        SB15.Z[SB15.Z < 1e-4] = 1e-4
        SB15.Z[SB15.Z > 0.03] = 0.03

        # go through each metallicity bin and add systems to the SFH until we have enough
        for i in range(len(target_counts)):
            if counts[i] == target_counts[i]:
                continue

            matches = (const.Z_BIN_EDGES[i] <= SB15.Z.value) & (SB15.Z.value < const.Z_BIN_EDGES[i+1])
            if matches.sum() == 0:
                continue
            elif matches.sum() > target_counts[i] - counts[i]:
                # if we have more matches than we need, randomly select a subset of the matches to add
                matches[np.random.choice(np.where(matches)[0], matches.sum() - (target_counts[i] - counts[i]), replace=False)] = False

            sfhs.append(SB15[matches])
            counts[i] = min(target_counts[i], counts[i] + matches.sum())

        print(f"    After iteration {rnd + 1}, we have {np.sum(counts)} systems in the right metallicity bins.")
        rnd += 1

    print(f"  [{time.time() - lap:03.1f}s] Finished sampling from SFH after {rnd} iterations. Sampled {np.sum(counts)} systems.")
    lap = time.time()

    # start a new population with the random sample
    p = cogsworth.pop.Population(
        len(rand_sample),
        # unused but required for cogsworth
        sfh_model=cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential(version='v2')),
        ini_file=f"/mnt/home/twagg/projects/frank-lisa/settings/{variation}.ini",
        # evolve through MW, only tracking end point, just 2 processes
        galactic_potential=gp.MilkyWayPotential(version='v2'),
        processes=2, store_entire_orbits=False,
        # make retries more often, but not quite so large a decrease in timestep
        # (avoids more failures without tanking runtime)
        orbit_integration_retry_settings={
            "timestep_multiplier": 0.25,
            "max_retries": 4
        },
        error_file_path=None
    )

    # stick together all of the samples galaxy SFHs
    p._initial_galaxy = cogsworth.sfh.concat(*sfhs)

    # reset the bin_nums for the random sample and kick info
    rand_sample.index = np.arange(len(rand_sample))
    rand_sample["bin_num"] = rand_sample.index.values
    rand_kicks.index = np.arange(len(rand_kicks)) // 2
    rand_kicks["bin_num"] = rand_kicks.index.values
    rand_initC.index = np.arange(len(rand_initC))
    rand_initC["bin_num"] = rand_initC.index.values

    if rand_initC_at_DCO is not None:
        rand_initC_at_DCO.index = np.arange(len(rand_initC_at_DCO))
        rand_initC_at_DCO["bin_num"] = rand_initC_at_DCO.index.values

    # these are all zero by definition, but we need to set them explicitly to avoid errors in cogsworth
    rand_kicks[['disrupted', 'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2']] = 0.0

    # save both to the population
    p._initial_binaries = rand_initC
    p._initial_binaries["MW_Z_weight"] = rand_sample["MW_Z_weight"].values
    p._initial_binaries["metallicity"] = rand_sample["metallicity"].values
    p._initial_binaries = p._initial_binaries.copy()

    p._kick_info = rand_kicks

    # bpp shouldn't have weights or metallicity and needs evol_type, let's just pick 15
    bpp = rand_sample.copy()
    bpp.drop(columns=["MW_Z_weight", "metallicity"], inplace=True)
    bpp["evol_type"] = 15
    p._bpp = bpp

    print(f"  [{time.time() - lap:03.1f}s] Finished setting up the population with {len(p)} systems.")
    lap = time.time()

    # step 1: mask out systems not yet formed DCOs
    # --------------------------------------------
    has_formed_dco = p.initial_galaxy.tau >= p.bpp["tphys"].values * u.Myr
    p_dco = p[has_formed_dco]

    # step 2: separate the RLOF systems from the pure GWs
    # ---------------------------------------------------

    # skip this if there's no RLOF time column, i.e. if the DCO type is not NSWD or BHWD
    if "rlof_time" in p_dco.bpp.columns:
        undergoing_rlof = p_dco.initial_galaxy.tau >= p_dco.bpp["rlof_time"].values * u.Myr
        recent_rlof = (
            undergoing_rlof
            & (p_dco.initial_galaxy.tau < p_dco.bpp["rlof_time"].values * u.Myr + 1 * u.Gyr)
        )

        # we only care about the recent RLOF systems, others have widened so much their SNR will be negligible
        p_rlof = p_dco[recent_rlof] if np.sum(recent_rlof) > 0 else None
        p_gw = p_dco[~undergoing_rlof]
    else:
        p_rlof = None
        p_gw = p_dco

    print(f"  [{time.time() - lap:03.1f}s] Found that {len(p_rlof) if p_rlof is not None else 0} systems are undergoing RLOF and {len(p_gw)} systems are pure GWs.")
    lap = time.time()

    # step 3: re-evolve RLOF systems with COSMIC to get their current orbital parameters
    # ----------------------------------------------------------------------------------
    if p_rlof is not None:
        if rand_initC_at_DCO is None:
            raise ValueError("RLOF systems require initC_at_DCO data to re-evolve with COSMIC, but no initC data was provided.")
        ibt = rand_initC_at_DCO.loc[p_rlof.bpp["bin_num"]].copy()
        ibt["tphysf"] = p_rlof.initial_galaxy.tau.to(u.Myr).value
        re_ev_bpp, _, _, _ = Evolve.evolve(ibt, nproc=2)

        # keep only the last row of the bpp (present state)
        re_ev_bpp = re_ev_bpp.drop_duplicates(subset="bin_num", keep="last")

        # copy over the cols COSMIC didn't produce
        extra_cols = ["weights", "remove_if_pessimistic", "rlof_time", "rlof_sep", "bin_num"]
        for col in extra_cols:
            re_ev_bpp[col] = p_rlof.bpp[col].values
        p_rlof._bpp = re_ev_bpp[p_rlof.bpp.columns]

        p_rlof = p_rlof[p_rlof.final_bpp["sep"] > 0]

    print(f"  [{time.time() - lap:03.1f}s] Evolved the {len(p_rlof) if p_rlof is not None else 0} RLOF systems to their current orbital parameters with COSMIC.")
    lap = time.time()


    # step 4: mask out the systems that are not inspiraling at present day
    # --------------------------------------------------------------------
    sources = p_gw.to_legwork_sources(distances=np.full(len(p_gw), 10.0) * u.pc)
    sources.n_proc = 2

    # first calculate the merger time approximately and jettison anything that merged over 100 Myr ago
    sources.get_merger_time(exact=False)
    is_inspiraling_approx = (
        p_gw.initial_galaxy.tau <= sources.t_merge
                                    + p_gw.bpp["tphys"].values * u.Myr
                                    + 100 * u.Myr
    )
    sources_maybe_insp = sources[is_inspiraling_approx]
    p_maybe_insp = p_gw[is_inspiraling_approx]

    # redo it with the real merger time
    sources_maybe_insp.get_merger_time(exact=True)
    is_inspiraling = (
        p_maybe_insp.initial_galaxy.tau <= sources_maybe_insp.t_merge
                                            + p_maybe_insp.bpp["tphys"].values * u.Myr
    )
    p_insp = p_maybe_insp[is_inspiraling]
    sources_insp = sources_maybe_insp[is_inspiraling]

    print(f"  [{time.time() - lap:03.1f}s] After first pass of GW systems, {len(p_insp)} systems are inspiraling at present day")
    lap = time.time()

    # step 5: mask out systems that have frequencies below 1e-6 Hz at present day
    # if not retaining the intrinsic population, hone down to just the systems that
    # are loud enough to be detectable by LISA at 10 pc
    sources_insp.evolve_sources(t_evol=p_insp.initial_galaxy.tau - p_insp.bpp["tphys"].values * u.Myr)

    mask = sources_insp.f_orb > 1e-6 * u.Hz
    if not retain_intrinsic:
        sources_insp.sc_params["t_obs"] = 10 * u.yr
        sources_insp.get_snr()
        mask &= (sources_insp.snr > 7)

    p_masked_no_rlof = p_insp[mask]
    sources_insp_masked = sources_insp[mask]
    _add_dummy_stats(p_masked_no_rlof)

    last_words = "have frequencies above 1e-6 Hz" if retain_intrinsic else "are loud enough to be detectable by LISA at 10 pc"
    print(f"  [{time.time() - lap:03.1f}s] After second pass, {len(p_masked_no_rlof)} systems {last_words}")
    lap = time.time()

    if p_rlof is not None:
        _add_dummy_stats(p_rlof)

        p_masked = cogsworth.pop.concat(p_masked_no_rlof, p_rlof)
        p_masked._final_bpp = None
        p_masked._disrupted = None
    else:
        p_masked = p_masked_no_rlof
    print(f"  [{time.time() - lap:03.1f}s] Combined the {len(p_masked_no_rlof)} GW systems and {len(p_rlof) if p_rlof is not None else 0} RLOF systems to get {len(p_masked)} total systems.")
    lap = time.time()

    # print(len(p_masked), "len p_masked before orbits")

    # final pass: evolve the loud systems through the Milky Way and calculate their SNRs at true distances
    p_masked.perform_galactic_evolution(progress_bar=False)
    print(f"  [{time.time() - lap:03.1f}s] Finished integrating the Galactic orbits of those systems through the Milky Way")
    lap = time.time()

    # print(len(p_masked), "len p_masked after orbits")

    sources_insp_masked = sources_insp[np.isin(p_insp.bin_nums, p_masked.bin_nums[p_masked.bin_nums <= max(p_masked_no_rlof.bin_nums)])]
    sources_insp_masked.sc_params["t_obs"] = 10 * u.yr

    # print(len(sources_insp_masked), "len sources_insp_masked before concat")

    # if there are RLOF systems, add them to the sources_insp_masked so we can calculate their SNRs too
    if p_rlof is not None:
        # mask the p_rlof to those that made it through orbit integration
        # print(len(p_rlof))
        p_rlof = p_rlof[np.isin(p_rlof.bin_nums, p_masked.bin_nums - (max(p_masked_no_rlof.bin_nums) + 1))]
        # print(len(p_rlof))
        sources_rlof = p_rlof.to_legwork_sources(distances=np.full(len(p_rlof), 10.0) * u.pc)
        # print(len(sources_rlof), (p_rlof.final_bpp["sep"] <= 0).sum())

        # append these sources to the main class
        for attr in ["m_1", "m_2", "dist", "ecc", "f_orb", "merged"]:
            # print(attr, len(getattr(sources_insp_masked, attr)), len(getattr(sources_rlof, attr)))
            setattr(sources_insp_masked, attr, np.concatenate([getattr(sources_insp_masked, attr),
                                                                getattr(sources_rlof, attr)]))
            # print(attr, len(getattr(sources_insp_masked, attr)))
        sources_insp_masked.t_merge = None
        sources_insp_masked.snr = None
        sources_insp_masked.max_snr_harmonic = None

        # print(len(sources_insp_masked), "len sources_insp_masked after concat")

    initial_distance = SkyCoord(
        x=p_masked.initial_galaxy.x, y=p_masked.initial_galaxy.y, z=p_masked.initial_galaxy.z,
        v_x=p_masked.initial_galaxy.v_x, v_y=p_masked.initial_galaxy.v_y, v_z=p_masked.initial_galaxy.v_z,
        representation_type="cartesian", unit=u.kpc, frame="galactocentric"
    ).icrs.distance

    instruments = ["LISA", "DECIGO"]
    positions = [("initial_pos", initial_distance),
                 ("final_pos", p_masked.get_final_mw_skycoord().icrs.distance)]
    mission_length = [("8yr", 8 * u.yr), ("4yr", 4 * u.yr)]
    snr_lims = [("7", 7), ("12", 12)]

    detectable_any_method = np.zeros(len(p_masked), dtype=bool)
    f_detects = {}
    weights_all = np.sum(p.bpp["weights"])

    for instrument in instruments:
        for pos_name, pos in positions:
            # print(pos_name, len(pos), len(p_masked), len(sources_insp_masked))
            for dur_name, dur in mission_length:
                sources_insp_masked.sc_params["instrument"] = instrument
                sources_insp_masked.sc_params["t_obs"] = dur
                sources_insp_masked.dist = pos
                snr = sources_insp_masked.get_snr()
                p_masked.bpp[f"snr_{instrument.lower()}_{dur_name}_{pos_name}"] = snr
                for snr_lim_name, snr_lim in snr_lims:
                    detectable_any_method |= (snr > snr_lim)
                    f_detects[f"{instrument.lower()}_{dur_name}_{pos_name}_SNRgt{snr_lim_name}"] = np.sum(p_masked.bpp[snr > snr_lim]["weights"]) / weights_all

    print(f"  [{time.time() - lap:03.1f}s] After final pass, {np.sum(detectable_any_method)} systems are detectable by LISA or DECIGO at 10 pc, either at their initial or final positions")
    lap = time.time()

    # throw out undetectable systems if we're not retaining the intrinsic population
    if not retain_intrinsic:
        if np.sum(detectable_any_method) == 0:
            print("  No systems are detectable by LISA or DECIGO at 10 pc, either at their initial or final positions. Returning an empty population.")
            p_masked = None
        else:
            p_masked = p_masked[detectable_any_method]

    # et voila, a detectable fraction for this Milky Way instance
    if p_masked is not None:
        _add_dummy_stats(p_masked)

    return f_detects, p_masked, weights_all


def _add_dummy_stats(p):
    p._mass_binaries = 0.0
    p._mass_singles = 0.0
    p._n_singles_req = 0
    p._n_bin_req = 0
    return p


def main():
    parser = argparse.ArgumentParser(description="Evolve a Milky Way instance and calculate the number of DCOs several times.")
    parser.add_argument("-d", "--dco_type", type=str, required=True, choices=["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"], help="Type of DCO to evolve.")
    parser.add_argument("-N", "--n_per_instance", type=int, default=500_000, help="Number of systems to sample per Milky Way instance.")
    parser.add_argument("-n", "--n_instances", type=int, default=10, help="Number of times to repeat the evolution.")
    parser.add_argument("-s", "--suffix", type=str, default="", help="Suffix to add to the output files.")
    parser.add_argument("-p", "--pessimistic", action="store_true", help="Use the pessimistic CE assumption when calculating the detectable fraction.")
    parser.add_argument("-r", "--retain_intrinsic", action="store_true", help="Retain the intrinsic population of DCOs, rather than just the detectable ones.")
    parser.add_argument("-v", "--variation", type=str, default="fiducial", help="Variation of the model to use (default: fiducial).")

    args = parser.parse_args()

    args.folder = os.path.join("/mnt/ceph/users/twagg/lisa-dcos", args.variation)
    args.output_folder = os.path.join(args.folder, "detection_files")

    os.makedirs(args.output_folder, exist_ok=True)

    full_start = time.time()
    all_formation_rows = pd.read_hdf(os.path.join(args.folder, f"{args.dco_type}_at_formation.h5"), key="formation_rows")
    print(f"Loaded {len(all_formation_rows)} formation rows in {time.time() - full_start:.2f} seconds.")

    lap = time.time()
    all_kick_infos = pd.read_hdf(os.path.join(args.folder, f"{args.dco_type}_at_formation.h5"), key="kick_info")
    print(f"Loaded {len(all_kick_infos)} kick infos in {time.time() - lap:.2f} seconds.")

    lap = time.time()
    all_initCs = load_initC(os.path.join(args.folder, f"{args.dco_type}_at_formation.h5"), key="initC", settings_key="initC_settings")
    print(f"Loaded {len(all_initCs)} initC in {time.time() - lap:.2f} seconds.")

    lap = time.time()
    with h5.File(os.path.join(args.folder, f"{args.dco_type}_at_formation.h5"), "r") as f:
        if "initC_at_DCO" in f:
            all_initC_at_DCOs = load_initC(os.path.join(args.folder, f"{args.dco_type}_at_formation.h5"), key="initC_at_DCO", settings_key="initC_at_DCO_settings")
        else:
            all_initC_at_DCOs = None
    print(f"Loaded {len(all_initC_at_DCOs) if all_initC_at_DCOs is not None else 0} initC_at_DCO in {time.time() - lap:.2f} seconds.")

    if args.pessimistic:
        print("Using pessimistic CE assumption: removing systems that would have been removed under this assumption.")
        all_formation_rows = all_formation_rows[~all_formation_rows["remove_if_pessimistic"]]

    all_formation_rows["MW_Z_weight"] = 0.0
    for Z in const.Z_BIN_CENTRES:
        all_formation_rows.loc[all_formation_rows["metallicity"] == Z, "MW_Z_weight"] = MW_weights[np.digitize(Z, const.Z_BIN_CENTRES) - 1]

    f_detect_dict = {}
    p_mws = []
    total_weight_list = []
    for inst in range(args.n_instances):
        print(f"Running Milky Way instance {inst + 1}/{args.n_instances}...")
        start = time.time()
        if args.retain_intrinsic:
            print("Retaining the intrinsic population of DCOs.")
        f_detect, p_mw, weights_all = evolve_milky_way_instance(
            all_formation_rows, all_kick_infos, all_initCs, all_initC_at_DCOs,
            n_per_instance=args.n_per_instance,
            oversample_factor=5,
            retain_intrinsic=args.retain_intrinsic,
            variation=args.variation
        )
        if p_mw is not None:
            p_mw.bpp["MW_instance"] = inst
            p_mws.append(p_mw)
        total_weight_list.append(weights_all)

        for key in f_detect.keys():
            if key not in f_detect_dict:
                f_detect_dict[key] = [f_detect[key]]
            else:
                f_detect_dict[key].append(f_detect[key])
        print(f"  Instance {inst + 1} finished in {time.time() - start:.2f} seconds. Detectable fraction for LISA final position: {f_detect['lisa_8yr_final_pos_SNRgt7']:.4e}")

    p_mw_all = cogsworth.pop.concat(*p_mws)
    f_detect_df = pd.DataFrame(f_detect_dict)
    print(f"Mean detectable fraction for LISA 8yr final pos, SNR > 7: {f_detect_df['lisa_8yr_final_pos_SNRgt7'].mean():.4e}")
    print(f"Mean detectable fraction for LISA 4yr final pos, SNR > 12: {f_detect_df['lisa_4yr_final_pos_SNRgt12'].mean():.4e}")

    sfh = cogsworth.sfh.StarFormationHistory()
    for var in ["_x", "_y", "_z", "_v_x", "_v_y", "_v_z", "_tau", "_Z"]:
        setattr(sfh, var, getattr(p_mw_all.initial_galaxy, var))
    p_mw_all._initial_galaxy = sfh

    pessimistic_str = "_pessimistic" if args.pessimistic else ""

    p_mw_all.save(os.path.join(args.output_folder, f"{args.dco_type}{pessimistic_str}_in_band_{args.suffix}.h5"), overwrite=True)
    f_detect_df.to_hdf(os.path.join(args.output_folder, f"{args.dco_type}{pessimistic_str}_f_detect_{args.suffix}.h5"), key="f_detect", mode="w")

    with h5.File(os.path.join(args.output_folder, f"{args.dco_type}{pessimistic_str}_f_detect_{args.suffix}.h5"), "a") as f:
        f.create_dataset("total_weights", data=np.array(total_weight_list))

    print(f"Saved detectable population and detectable fractions to {args.output_folder}.")
    print(f"Total time for {args.n_instances} instances: {time.time() - full_start:.2f} seconds.\n\n\n")


if __name__ == "__main__":
    main()

# python sample_detectable_population.py -f /mnt/ceph/users/twagg/lisa-dcos/fiducial/ -d NSWD -N 500000 -n 1 -o /mnt/ceph/users/twagg/lisa-dcos/fiducial/detection_files -s test
# python sample_detectable_population.py -d NSWD -N 500000 -n 1  -s test -v fiducial