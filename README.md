# Charting the Galactic Underworld II: Strategies for Identifying BHs and NSs with LISA

## About the paper

LISA should resolve hundreds of Galactic double compact objects (DCOs) that contain a black hole (BH) or neutron star (NS). The hard part is picking them out from the much larger population of double white dwarfs (WDWDs).

In this work we:

1. Synthesise large populations of BHBH, BHNS, NSNS, BHWD and NSWD binaries with [COSMIC](https://cosmic-popsynth.github.io/COSMIC/) (v4.2.1), using its implementation of the STROOPWAFEL adaptive importance sampling algorithm (50 metallicity bins, 2 million binaries per bin, per DCO type).
2. Place them in the Milky Way with [cogsworth](https://cogsworth.readthedocs.io/) (v4.0.2), using the Sanders & Binney (2015) star formation history, and integrate their Galactic orbits in the [gala](http://gala.adrian.pw/) `MilkyWayPotential` (v2), accounting for SN kicks.
3. Evolve each binary to the present day and compute LISA (and DECIGO) SNRs with [LEGWORK](https://legwork.readthedocs.io/) (v1.0.0), for 2560 Milky Way realisations per DCO type and model.
4. Repeat for a set of binary physics variations (see below).

Our main findings:

- SN kicks substantially increase the scale height of BH and NS binaries, but LISA detection rates are mostly insensitive to this change in position.
- Chirp mass alone identifies only ~50-60% of BHBHs and BHNSs and ~40% of NSNSs. Adding orbital frequency recovers 90-100% of BHBHs and BHNSs. Adding eccentricity raises NSNS recovery to ~60-70%.
- Distance from the Galactic plane adds little, though it picks up a few NSWDs that would otherwise be missed.

## Model variations

Each variation has a COSMIC settings file in [settings/](settings/):

| Settings file | Paper label | Description |
|:--|:--|:--|
| `fiducial.ini` | Fiducial / Optimistic CE | Fiducial model |
| `qcflag2.ini` | C14 $q_{\rm crit}$ | Claeys et al. (2014) critical mass ratios for mass-transfer stability |
| `alpha_low.ini` | $\alpha_{\rm CE} = 0.1$ | CE ejection efficiency reduced 10x |
| `alpha_high.ini` | $\alpha_{\rm CE} = 10$ | CE ejection efficiency increased 10x |
| `maltsev.ini` | M25 remnants | Maltsev et al. (2025) / Willcox et al. (2025) remnant masses |
| `no_ecsn.ini` | No ECSN | Electron-capture supernovae turned off |

The simulations themselves use the optimistic common-envelope treatment. The "pessimistic" treatment, used by the fiducial model and every variation, is applied afterwards by removing binaries that survived a CE initiated by a Hertzsprung-gap donor (the `--pessimistic` flag in the scripts, and the `_pessimistic` suffix in file names). The fiducial run without this cut is the "Optimistic CE" model.

## Repository layout

```
├── settings/    COSMIC .ini files for each model variation
├── src/         Pipeline scripts (population synthesis -> Milky Way sampling -> post-processing)
├── slurm/       Generators for the Slurm/disBatch job scripts that run the pipeline on Rusty
├── helpers/     Shared Python modules used by the scripts and notebooks
├── notebooks/   Analysis and figure-making notebooks
├── plots/       Figures produced by the notebooks
└── data/        Small cached inputs (Milky Way metallicity weights)
```

### `src/`: the pipeline

Run in roughly this order:

| Script | Purpose |
|:--|:--|
| `create_dco_population.py` | Run STROOPWAFEL sampling in COSMIC for a single DCO type, metallicity and settings file |
| `postprocess_dcos.py` (`postprocess.sh`) | Collect the STROOPWAFEL outputs across metallicities and extract the DCO formation rows and initial conditions |
| `sample_detectable_population.py` | Draw Milky Way realisations with cogsworth, integrate Galactic orbits, evolve to present day, compute SNRs and $f_{\rm detect}$ |
| `concat_detectable_pops.py` (`concat.sh`) | Merge the per-job detectable populations and $f_{\rm detect}$ files |
| `norm.py` | Compute $N_{\rm target, MW}$, the normalisation to the Milky Way's total star-forming mass |
| `calculate_distinguishers.py` | Apply LISA measurement uncertainties and compute how well each property separates DCOs from WDWDs |
| `check_files.py`, `man_fix.sh` | Utilities for checking outputs and finding missing job outputs |

### `slurm/`

- `stroopwafel_runs/generate_slurm.py` writes one Slurm array job per DCO type and variation, one task per metallicity bin, each running `create_dco_population.py`.
- `detectable_runs/generate_slurm.py` writes disBatch jobs that run `sample_detectable_population.py` many times to build up the 2560 Milky Way realisations.

### `helpers/`

| Module | Contents |
|:--|:--|
| `const.py` | Metallicity bins, minimum primary masses, DCO types, colours and list of variations |
| `initial_dists.py` | IMF and binary-fraction (Offner et al. 2023) distributions |
| `truncation.py` | Mass fraction of the sampled population relative to the full stellar population, used in normalisation |
| `sfh.py` | Modified Sanders & Binney (2015) star formation history |
| `uncertainties.py` | LISA measurement uncertainties on chirp mass, eccentricity, frequency and position |
| `detections.py` | Loading and summarising detection rates |
| `plotting.py` | Shared plotting functions and style |

### `notebooks/`

| Notebook | Contents |
|:--|:--|
| `detectable.ipynb` | Positions, kick magnitudes and intrinsic properties of detectable DCOs (`dco_positions.pdf`, `kick_magnitudes_two_panels.pdf`, `dco_properties.pdf`) |
| `rates.ipynb` | Detection rates for each model and their tables (`n_detect.pdf`) |
| `uncertainties.ipynb` | Properties after applying measurement uncertainties (`dco_measured_properties.pdf`) |
| `distinguish.ipynb` | How well each property identifies DCOs (`distinguishers.pdf`) |
| `formation.ipynb`, `rlof.ipynb`, `re-evolution.ipynb` | Formation channels, mass-transferring NSWDs/BHWDs, and re-evolving systems with COSMIC |
| `lisa_numbers.ipynb` | Numbers quoted in the text |
| `zenodo.ipynb` | Packages the simulation outputs for the Zenodo data release |
| `scratch/` | Exploratory and one-off notebooks |

## Data

The simulated populations of detectable DCOs, with their COSMIC initial conditions, the detectable fractions for each Milky Way realisation and the normalisations, [are on Zenodo](https://zenodo.org/records/23520104).

Many scripts and notebooks have hard-coded paths to the full simulation outputs on Flatiron's Ceph storage (`/mnt/ceph/users/twagg/lisa-dcos`). Change these to point at your own copy of the data.

## Citation

If you use this code please cite the paper, if you use the data please cite both the paper and the Zenodo record. Feel free to reach out to me if I can help you use this data for a project. Thanks!
