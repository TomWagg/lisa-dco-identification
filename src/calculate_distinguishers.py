import os
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import time

import legwork as lw
import gala.potential as gp
import cogsworth

import sys
sys.path.append("../helpers")

import const
import plotting
import uncertainties as unc


def calculate_distinguishers(variation, counts, pess=True, dur=8, snr_lim=12, n_boot=5000):
    print(f"Calculating distinguishers for {variation}")
    pess_str = "_pessimistic" if pess and variation == "fiducial" else ""
    pops = {
        dco_type: cogsworth.pop.load(f"/mnt/ceph/users/twagg/lisa-dcos/{variation}/{dco_type}{pess_str}_in_band.h5") for dco_type in const.DCO_TYPES
    }
    print("Loaded populations")

    detectable_pops = {}
    for dco_type in const.DCO_TYPES:
        pop = pops[dco_type]
        pop.final_pos, pop.final_vel
        
        mask = (
            (pops[dco_type].bpp[f"snr_lisa_{dur}yr_final_pos"] > snr_lim)
            | (pops[dco_type].bpp[f"snr_lisa_{dur}yr_initial_pos"] > snr_lim)
            | (pops[dco_type].bpp[f"snr_decigo_{dur}yr_final_pos"] > snr_lim)
            | (pops[dco_type].bpp[f"snr_decigo_{dur}yr_initial_pos"] > snr_lim)
        )
        detectable_pops[dco_type] = pops[dco_type][mask]
        print(f"Detectable {dco_type} loaded: {len(detectable_pops[dco_type])}")

    lisa_pops = {dco_type: pops[dco_type][pops[dco_type].bpp[f"snr_lisa_{dur}yr_final_pos"] > snr_lim] for dco_type in const.DCO_TYPES}
    print("Masked LISA populations")

    detectable_sources = {}
    for dco_type in const.DCO_TYPES:
        path = f"/mnt/ceph/users/twagg/lisa-dcos/{variation}/{dco_type}{pess_str}_detectable_sources.h5"
        if os.path.isfile(path):
            detectable_sources[dco_type] = lw.source.Source.from_file(path)
        else:
            # turn them into a LEGWORK source class
            detectable_sources[dco_type] = detectable_pops[dco_type].to_legwork_sources(assume_mw_galactocentric=True)
            detectable_sources[dco_type].n_proc = 30

            # find those that haven't reached RLOF yet evolve them via GWs until present day
            if "rlof_time" not in detectable_pops[dco_type].bpp.columns:
                detectable_pops[dco_type].bpp["rlof_time"] = 1e20

            reached_rlof = detectable_pops[dco_type].initial_galaxy.tau > detectable_pops[dco_type].bpp["rlof_time"].values * u.Myr

            non_rlof_sources = detectable_sources[dco_type][~reached_rlof]
            non_rlof_sources.evolve_sources(t_evol=detectable_pops[dco_type].initial_galaxy.tau[~reached_rlof] - detectable_pops[dco_type].bpp["tphys"].values[~reached_rlof] * u.Myr)

            # recombine into a single source class
            detectable_sources[dco_type].m_1[~reached_rlof] = non_rlof_sources.m_1
            detectable_sources[dco_type].m_2[~reached_rlof] = non_rlof_sources.m_2
            detectable_sources[dco_type].dist[~reached_rlof] = non_rlof_sources.dist
            detectable_sources[dco_type].ecc[~reached_rlof] = non_rlof_sources.ecc
            detectable_sources[dco_type].f_orb[~reached_rlof] = non_rlof_sources.f_orb
            detectable_sources[dco_type].merged[~reached_rlof] = non_rlof_sources.merged

            detectable_sources[dco_type].save(path, overwrite=True)
        print(f"Detectable sources for {dco_type} loaded/computed: {len(detectable_sources[dco_type])}")

    lisa_sources = {dco_type: detectable_sources[dco_type][detectable_pops[dco_type].bpp[f"snr_lisa_{dur}yr_final_pos"] > snr_lim] for dco_type in const.DCO_TYPES}


    wdwd_dist = cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential(version="v2"))
    wdwd_dist.sample(500_000)
    print("Sampled WDWD distribution")

    unc_data = unc.get_unc_data(lisa_sources, lisa_pops, dur=dur, snr_lim=snr_lim, snr_harm_lim=snr_lim)
    print(f"Calculated uncertainty data for {dco_type}")

    fig, axes, height_where_exceeds_wdwds = plotting.four_panel_uncertainties(
        lisa_sources, lisa_pops, unc_data, wdwd_dist, detectable_pops, counts,
        n_boot=n_boot, save=f"../plots/dco_measured_properties_{variation}{pess_str}.pdf", show=False
    )
    plt.close()
    print("Plotted four-panel uncertainties")

    distinguishers = unc.get_wdwd_distinguishing_factors(
        lisa_sources, lisa_pops, unc_data, height_where_exceeds_wdwds,
        max_wdwd_mass=lw.utils.chirp_mass(1.44, 1.44) * u.Msun, min_wdwd_forb=3e-4 * u.Hz
    )
    print("Calculated WDWD distinguishing factors")

    file_path = f"/mnt/ceph/users/twagg/lisa-dcos/{variation}/uncertainties{pess_str}.h5"
    for dco_type in const.DCO_TYPES:
        unc_df = pd.DataFrame(unc_data[dco_type])
        unc_df.to_hdf(file_path, key=f"uncertainties/{dco_type}", mode="a", format="table", data_columns=True)
    distinguishers.to_hdf(file_path, key="distinguishers", mode="a", format="table", data_columns=True)

    print(f"Saved uncertainties and distinguishers to {file_path}")


def main():
    counts = {
        # "fiducial": {"BHBH": 12, "BHNS": 6, "BHWD": 2, "NSNS": 2, "NSWD": 99},
        # "optimistic": {"BHBH": 35, "BHNS": 14, "BHWD": 9, "NSNS": 5, "NSWD": 161},
        # "alpha_low": {"BHBH": 13, "BHNS": 4, "BHWD": 2, "NSNS": 2, "NSWD": 2},
        "alpha_high": {"BHBH": 2, "BHNS": 5, "BHWD": 0, "NSNS": 6, "NSWD": 22},
        "maltsev": {"BHBH": 1, "BHNS": 4, "BHWD": 0, "NSNS": 0, "NSWD": 104},
        "no_ecsn": {"BHBH": 12, "BHNS": 6, "BHWD": 2, "NSNS": 2, "NSWD": 106},
        "qcflag2": {"BHBH": 12, "BHNS": 7, "BHWD": 2, "NSNS": 2, "NSWD": 63},
    }

    start = time.time()
    for variation in counts.keys():
        lap = time.time()
        cnts = counts[variation]
        if variation == "optimistic":
            pess = False
            variation = "fiducial"
        else:
            pess = True
        calculate_distinguishers(variation, cnts, pess=pess, dur=8, snr_lim=12, n_boot=5000)
        print(f"Finished {variation} in {time.time() - lap:.2f} seconds")
    print(f"Finished all variations in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()