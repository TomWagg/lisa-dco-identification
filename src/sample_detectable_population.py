import os.path
import argparse
import time

import pandas as pd
import astropy.units as u
import numpy as np

import sys
sys.path.append("../helpers")
import const

import gala.potential as gp
import cogsworth


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



def evolve_milky_way_instance(all_formation_rows, all_kick_infos, n_per_instance=500_000):
    lap = time.time()

    # draw a random sample, weighted by the Milky Way metallicity distribution
    rand_sample = all_formation_rows.sample(n_per_instance, weights="MW_Z_weight", replace=True)
    rand_kicks = all_kick_infos.loc[rand_sample.index]

    print(f"  [{time.time() - lap:03.1f}s] Sampled {n_per_instance} systems from the formation rows")
    lap = time.time()

    # work out the target number of systems in each metallicity bin, for galaxy sampling
    target_counts, _ = np.histogram(rand_sample["metallicity"].values, bins=const.Z_BIN_EDGES)
    sample_size = target_counts.sum() * 8

    # loop variables
    counts, rnd, sfhs = np.zeros_like(target_counts), 0, []

    print(f"  [{time.time() - lap:03.1f}s] Starting to sample from the SFH to get {target_counts.sum()} systems in the right metallicity bins...")
    lap = time.time()

    # continue until we have enough systems in each metallicity bin
    while np.sum(counts) < np.sum(target_counts):
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
        ini_file="/mnt/home/twagg/projects/frank-lisa/src/params.ini",
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

    # these are all zero by definition, but we need to set them explicitly to avoid errors in cogsworth
    rand_kicks[['disrupted', 'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2']] = 0.0

    # save both to the population
    p._initial_binaries = rand_sample
    p._kick_info = rand_kicks

    # bpp shouldn't have weights or metallicity and needs evol_type, let's just pick 15
    bpp = rand_sample.copy()
    bpp.drop(columns=["MW_Z_weight", "metallicity"], inplace=True)
    bpp["evol_type"] = 15
    p._bpp = bpp

    print(f"  [{time.time() - lap:03.1f}s] Finished setting up the population with {len(p)} systems. Starting LEGWORK evolution...")
    lap = time.time()

    # first pass: mask out systems that have merged or not yet formed DCOs
    sources = p.to_legwork_sources(distances=np.full(len(p), 8.0) * u.kpc)
    sources.update_sc_params({
        "t_obs": 10 * u.yr
    })
    sources.get_merger_time(exact=False)
    is_inspiraling = (
        (p.initial_galaxy.tau >= p.bpp["tphys"].values * u.Myr) &                   # has formed a BHBH
        (p.initial_galaxy.tau <= sources.t_merge + p.bpp["tphys"].values * u.Myr)   # but hasn't merged yet
    )
    p_insp = p[is_inspiraling]

    print(f"  [{time.time() - lap:03.1f}s] After first pass, {len(p_insp)} systems are inspiraling at present day")
    lap = time.time()

    # second pass: mask out systems that are not detectable by LISA at 10 parsecs before integrating orbits
    sources_insp = p_insp.to_legwork_sources(distances=np.full(len(p_insp), 10.0) * u.pc)
    sources_insp.update_sc_params({
        "t_obs": 10 * u.yr
    })
    sources_insp.evolve_sources(t_evol=p_insp.initial_galaxy.tau)
    sources_insp.get_snr()
    p_insp_loud = p_insp[sources_insp.snr > 7]

    print(f"  [{time.time() - lap:03.1f}s] After second pass, {len(p_insp_loud)} systems are loud enough to be detectable by LISA at 10 pc")
    lap = time.time()

    # final pass: evolve the loud systems through the Milky Way and calculate their SNRs at true distances
    p_insp_loud.perform_galactic_evolution(progress_bar=False)
    sources_insp_loud = p_insp_loud.to_legwork_sources(assume_mw_galactocentric=True)
    sources_insp_loud.update_sc_params({
        "t_obs": 10 * u.yr
    })
    sources_insp_loud.evolve_sources(t_evol=p_insp_loud.initial_galaxy.tau)
    sources_insp_loud.get_snr()

    print(f"  [{time.time() - lap:03.1f}s] After final pass, {np.sum(sources_insp_loud.snr > 7)} systems are detectable by LISA at their true distances")
    lap = time.time()

    # sum up the weights for the detectable systems and the total population
    weights_detect = np.sum(p_insp_loud.bpp[sources_insp_loud.snr > 7]["weights"])
    weights_all = np.sum(p.bpp["weights"])

    # et voila, a detectable fraction for this Milky Way instance
    f_detect = weights_detect / weights_all
    p_detect = p_insp_loud[sources_insp_loud.snr > 7]
    p_detect._mass_binaries = 0.0
    p_detect._mass_singles = 0.0
    p_detect._n_singles_req = 0
    p_detect._n_bin_req = 0

    return f_detect, p_detect


def main():
    parser = argparse.ArgumentParser(description="Evolve a Milky Way instance and calculate the number of DCOs several times.")
    parser.add_argument("-f", "--folder", type=str, required=True, help="Path to the folder containing the formation rows and kick info.")
    parser.add_argument("-d", "--dco_type", type=str, required=True, choices=["NSWD", "NSNS", "BHWD", "BHNS", "BHBH"], help="Type of DCO to evolve.")
    parser.add_argument("-N", "--n_per_instance", type=int, default=500_000, help="Number of systems to sample per Milky Way instance.")
    parser.add_argument("-n", "--n_instances", type=int, default=10, help="Number of times to repeat the evolution.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the folder where the output files will be saved.")
    parser.add_argument("-s", "--suffix", type=str, default="", help="Suffix to add to the output files.")

    args = parser.parse_args()

    full_start = time.time()

    all_formation_rows = pd.read_hdf(os.path.join(args.folder, f"{args.dco_type}_formation_rows.h5"), key="formation_rows")
    all_kick_infos = pd.read_hdf(os.path.join(args.folder, f"{args.dco_type}_kick_info.h5"), key="kick_info")

    all_formation_rows["MW_Z_weight"] = 0.0
    for Z in const.Z_BIN_CENTRES:
        all_formation_rows.loc[all_formation_rows["metallicity"] == Z, "MW_Z_weight"] = const.MW_weights[np.digitize(Z, const.Z_BIN_CENTRES) - 1]

    f_detects = []
    p_detects = []
    for inst in range(args.n_instances):
        print(f"Running Milky Way instance {inst + 1}/{args.n_instances}...")
        start = time.time()
        f_detect, p_detect = evolve_milky_way_instance(
            all_formation_rows, all_kick_infos, n_per_instance=args.n_per_instance
        )
        f_detects.append(f_detect)
        p_detects.append(p_detect)
        print(f"  Instance {inst + 1} finished in {time.time() - start:.2f} seconds. Detectable fraction: {f_detect:.4e}")

    p_detect_all = cogsworth.pop.concat(*p_detects)
    f_detect_mean = np.mean(f_detects)
    f_detect_std = np.std(f_detects)
    print(f"Mean detectable fraction: {f_detect_mean:.4e} ± {f_detect_std:.4e}")

    sfh = cogsworth.sfh.StarFormationHistory()
    for var in ["_x", "_y", "_z", "_v_x", "_v_y", "_v_z", "_tau", "_Z"]:
        setattr(sfh, var, getattr(p_detect_all.initial_galaxy, var))
    p_detect_all._initial_galaxy = sfh

    p_detect_all.save(os.path.join(args.output_folder, f"{args.dco_type}_detectable_{args.suffix}.h5"), overwrite=True)
    np.save(os.path.join(args.output_folder, f"{args.dco_type}_f_detect_{args.suffix}.npy"), np.array(f_detects))

    print(f"Saved detectable population and detectable fractions to {args.output_folder}.")
    print(f"Total time for {args.n_instances} instances: {time.time() - full_start:.2f} seconds.\n\n\n")


if __name__ == "__main__":
    main()