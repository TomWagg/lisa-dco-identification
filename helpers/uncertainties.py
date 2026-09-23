import numpy as np
import astropy.units as u
import legwork as lw

import const
import pandas as pd


def get_ecc_uncertainty_stats(sources, harmonic_threshold=1):

    detectable_harmonics = np.zeros(len(sources)).astype(int)
    snr_uncertainty = np.zeros(len(sources))
    max_harmonics = np.zeros(len(sources)).astype(int)
    snr_ratio = np.zeros(len(sources))

    harmonics_required = sources.harmonics_required(sources.ecc)

    harmonic_groups = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
    for lower, upper in harmonic_groups:
        match = np.logical_and(harmonics_required > lower, harmonics_required <= upper)
        if match.any():
            snr_n_2 = lw.snr.snr_ecc_stationary(
                m_c=sources.m_c[match],
                f_orb=sources.f_orb[match],
                ecc=sources.ecc[match],
                dist=sources.dist[match],
                t_obs=4 * u.yr,
                harmonics_required=upper,
                interpolated_g=sources.g,
                interpolated_sc=sources.sc,
                ret_snr2_by_harmonic=True
            )

            # count harmonics above threshold
            detectable_harmonics[match] = (snr_n_2**0.5 > harmonic_threshold).astype(int).sum(axis=1)

            max_harmonics[match] = np.argmax(snr_n_2, axis=1) + 1

            # get the top two harmonics and sum them to get uncertainty
            top_snrs = np.sort(snr_n_2**(0.5), axis=1)[:, -2:]
            # replace any zeros with 1e-10 to avoid division by zero
            top_snrs[top_snrs == 0] = 1e-10
            snr_uncertainty[match] = 1 / top_snrs[:, 0] + 1 / top_snrs[:, -1]
            snr_ratio[match] = top_snrs[:, 0] / top_snrs[:, 1]

    return snr_uncertainty, snr_ratio, detectable_harmonics, max_harmonics

def sky_localisation(snr, fGW, L=2*u.AU):
    sigma_theta = 16.6 * (7 / snr) * (5e-4 * u.Hz / fGW) * (2 * u.AU / L) * u.deg
    return sigma_theta.to(u.deg)

def get_f_orb_uncertainty(snr, t_obs, f_orb):
    return (4 * np.sqrt(3) / np.pi / (snr * t_obs) / f_orb).decompose()

def get_f_orb_dot_uncertainty(snr, t_obs, f_orb_dot):
    return (6 * np.sqrt(5) / np.pi / (snr * t_obs**2) / f_orb_dot).decompose()

def get_Fprime_over_F(e):
    return e * (1256 + 1608 * e**2 + 111 * e**4) / (96 + 196 * e**2 - 255 * e**4 - 37 * e**6)

def get_m_c_uncertainty(f_orb, f_orb_dot, ecc, ecc_uncertainty, snr, t_obs):
    f_orb_uncertainty = get_f_orb_uncertainty(snr, t_obs, f_orb)
    f_orb_dot_uncertainty = get_f_orb_dot_uncertainty(snr, t_obs, f_orb_dot)

    return 11 / 5 * f_orb_uncertainty \
        + 3 / 5 * f_orb_dot_uncertainty \
        + 3 / 5 * get_Fprime_over_F(ecc) * ecc_uncertainty

def get_D_uncertainty(snr, f_dom, m_c, t_obs):
    return 0.2 * (snr / 10)**-1 * np.maximum(1, (f_dom / (1.4e-3 * u.Hz))**(-11/3) * (m_c / u.Msun)**(-5/3) * (t_obs / (10 * u.yr))**(-2))

def get_unc_data(lisa_sources, lisa_pops, dur=8, snr_lim=12, snr_harm_lim=12):
    unc_data = {}
    for dco_type in const.DCO_TYPES:
        lisa_sources[dco_type].n_proc = 30
        if lisa_sources[dco_type].snr is None:
            lisa_sources[dco_type].get_snr(verbose=True)
        snr_uncertainty, snr_ratio, detectable_harmonics, max_harmonics = get_ecc_uncertainty_stats(lisa_sources[dco_type], harmonic_threshold=snr_harm_lim)
        f_orb_dot = lw.utils.fn_dot(lisa_sources[dco_type].m_c, lisa_sources[dco_type].f_orb, lisa_sources[dco_type].ecc, n=1)
        delta_m_c_over_m_c = get_m_c_uncertainty(
            f_orb=lisa_sources[dco_type].f_orb, f_orb_dot=f_orb_dot, ecc=lisa_sources[dco_type].ecc,
            ecc_uncertainty=snr_uncertainty, snr=lisa_sources[dco_type].snr, t_obs=dur * u.yr
        )
        sigma_theta = sky_localisation(
            lisa_sources[dco_type].snr,
            lisa_sources[dco_type].f_orb * lisa_sources[dco_type].max_snr_harmonic
        )
        delta_d_over_d = get_D_uncertainty(
            snr=lisa_sources[dco_type].snr,
            f_dom=lisa_sources[dco_type].f_orb * lisa_sources[dco_type].max_snr_harmonic,
            m_c=lisa_sources[dco_type].m_c,
            t_obs=dur * u.yr
        )

        coords = lisa_pops[dco_type].get_final_mw_skycoord().galactic
        sigma_z = lisa_sources[dco_type].dist * np.sqrt(
            np.sin(coords.b.rad)**2 * delta_d_over_d**2 + np.cos(coords.b.rad)**2 * sigma_theta.to(u.rad).value**2
        )

        delta_forb = get_f_orb_uncertainty(
            snr=lisa_sources[dco_type].snr, t_obs=dur * u.yr, f_orb=lisa_sources[dco_type].f_orb
        ) * lisa_sources[dco_type].f_orb

        unc_data[dco_type] = {
            "snr_uncertainty": snr_uncertainty,
            "snr_ratio": snr_ratio,
            "detectable_harmonics": detectable_harmonics,
            "max_harmonics": max_harmonics,
            "delta_m_c_over_m_c": delta_m_c_over_m_c,
            "sigma_theta": sigma_theta,
            "delta_d_over_d": delta_d_over_d,
            "sigma_z": sigma_z,
            "delta_forb": delta_forb
        }
    return unc_data

def min_detectable_ecc(snr, f, f_min=0.5 * u.mHz, f_max=10 * u.mHz):
    min_det = (1 / snr**1.54 + 1 / snr) * (1.08 + 0.87 * np.arctan(1.08 * (f.to(u.mHz).value - 2.13)) - 0.55 * np.arctan(2.08 * (f.to(u.mHz).value - 1.22)))
    min_det[(f < f_min) | (f > f_max)] = 1
    return min_det

def get_wdwd_distinguishing_factors(
        lisa_sources, lisa_pops, unc_data, height_where_exceeds_wdwds,
        max_wdwd_mass=lw.utils.chirp_mass(1.44, 1.44) * u.Msun, min_wdwd_forb=3e-4 * u.Hz
    ):
    df_data = {
        "too_low_freq": [],
        "too_massive": [],
        "measureable_eccentricity": [],
        "too_far_from_plane": [],
        "m_forb": [],
        "m_forb_ecc": [],
        "m_forb_ecc_z": []
    }
    
    for dco_type in const.DCO_TYPES:
        w = lisa_pops[dco_type].bpp["weights"]

        too_massive = (lisa_sources[dco_type].m_c - unc_data[dco_type]["delta_m_c_over_m_c"] * lisa_sources[dco_type].m_c) > max_wdwd_mass
        too_low_freq = (lisa_sources[dco_type].f_orb + unc_data[dco_type]["delta_forb"]) < min_wdwd_forb
        at_least_two_harmonics = unc_data[dco_type]["detectable_harmonics"] >= 2
        ecc_detectable = lisa_sources[dco_type].ecc >= min_detectable_ecc(lisa_sources[dco_type].snr, lisa_sources[dco_type].f_orb * lisa_sources[dco_type].max_snr_harmonic)
        measureable_eccentricity = at_least_two_harmonics | ecc_detectable
        too_far_from_plane = np.abs(lisa_pops[dco_type].final_pos[:, 2]) - unc_data[dco_type]["sigma_z"] > height_where_exceeds_wdwds[dco_type] * u.kpc

        df_data["too_massive"].append(w[too_massive].sum() / w.sum())
        df_data["too_low_freq"].append(w[too_low_freq].sum() / w.sum())
        df_data["measureable_eccentricity"].append(w[measureable_eccentricity].sum() / w.sum())
        df_data["too_far_from_plane"].append(w[too_far_from_plane].sum() / w.sum())
        df_data["m_forb"].append(w[too_massive | too_low_freq].sum() / w.sum())
        df_data["m_forb_ecc"].append(w[too_massive | too_low_freq | measureable_eccentricity].sum() / w.sum())
        df_data["m_forb_ecc_z"].append(w[too_massive | too_low_freq | measureable_eccentricity | too_far_from_plane].sum() / w.sum())

    for k in df_data:
        df_data[k] = np.round(df_data[k], 3)

    df = pd.DataFrame(df_data, index=const.DCO_TYPES)

    return df
