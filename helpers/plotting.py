import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import legwork as lw

from copy import copy
import const
import uncertainties as unc


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': 0.7*fs,
          'legend.title_fontsize': 0.8*fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


class LogDecadeMinorLocator(mticker.Locator):
    """minor tick locator for axes holding log10(x) data, placing ticks at 2-9 in each decade"""
    def __call__(self):
        lo, hi = sorted(self.axis.get_view_interval())
        decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
        return (decades[:, None] + np.log10(np.arange(2, 10))[None, :]).ravel()


def fake_log_axis(ax, axis="x"):
    """make an axis of log10(x) data read like a standard matplotlib log axis

    Parameters
    ----------
    ax : `matplotlib Axis`
        Axis to adjust
    axis : `str`, optional
        Which axis to adjust, either "x" or "y", by default "x"

    Returns
    -------
    ax : `matplotlib Axis`
        Adjusted axis
    """
    a = ax.xaxis if axis == "x" else ax.yaxis

    a.set_major_locator(mticker.MultipleLocator(1))
    _log_fmt = mticker.LogFormatterSciNotation()
    _log_fmt.set_axis(a)
    a.set_major_formatter(mticker.FuncFormatter(lambda x, pos: _log_fmt(10**x, pos)))

    a.set_minor_locator(LogDecadeMinorLocator())
    a.set_minor_formatter(mticker.NullFormatter())

    # match the tick lengths matplotlib uses for a real log axis
    ax.tick_params(axis=axis, which="major", length=6)
    ax.tick_params(axis=axis, which="minor", length=3)

    return ax

class MirroredKDE(gaussian_kde):
    """ KDE class that mirrors data at boundaries to account for bounded support """

    def __init__(self, data, weights=None, lower_bound=None, upper_bound=None,
                 bw_method=None, bw_adjust=None):
        """ instantiate class in similar way to scipy but with some additions """
        super().__init__(data, weights=weights, bw_method=bw_method)

        # also store the lower and upper bounds
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # allow adjustment of the default bandwidth similar to seaborn
        if bw_adjust is not None:
            self.set_bandwidth(self.factor * bw_adjust)

    def evaluate(self, x_vals=None, x_min=None, x_max=None, x_count=200):
        """ evaluate the kde taking into account the boundaries """

        # only return x_vals when they aren't supplied
        return_x_vals = x_vals is None

        if x_vals is None:
            if x_min is None:
                x_min = np.min(self.dataset)
            if x_max is None:
                x_max = np.max(self.dataset)
            x_vals = np.linspace(x_min, x_max, x_count)

        # make a copy of the data before I mirror anything
        unmirrored_x_vals = np.copy(x_vals)

        # evaluate the kde at the original x values
        kde_vals = super().evaluate(x_vals)

        # if either bound is present then mirror the data and
        # add the evaluated kde for the mirrored data to the original
        if self._lower_bound is not None:
            x_vals = 2.0 * self._lower_bound - x_vals
            kde_vals += super().evaluate(x_vals)
            x_vals = unmirrored_x_vals

        if self._upper_bound is not None:
            x_vals = 2.0 * self._upper_bound - x_vals
            kde_vals += super().evaluate(x_vals)
            x_vals = unmirrored_x_vals

        if return_x_vals:
            return x_vals, kde_vals
        else:
            return kde_vals


def bootstrapped_kde(variable, weights, ax, seeds=None, bw_adjust=None, normalisation=1,
                     lower_bound=None, upper_bound=None,
                     bootstraps=200, x_min=None, x_max=None, x_count=200, log_scale=(False, False),
                     color="tab:blue", label=None, **kwargs):
    """Create a bootstrapped weighted KDE plot.

    Parameters
    ----------
    variable : `float/array`
        Variable that you want to make a KDE of.
    weights : 'float/array'
        Weights associated with each variable (see all to 1 for unweighted)
    seeds : `int/array`
        Seeds that make the binaries in COMPAS
    ax : `matplotlib Axis`
        Axis on which to plot
    bw_adjust : `float`, optional
        Factor by which to adjust the bandwidth, by default None
    bootstraps : `int`, optional
        How many bootstraps to do, by default 200
    x_count : `int`, optional
        How many x values to evaluate at, by default 500
    log_scale : `tuple`, optional
        Whether each axis should be log scaled, by default (False, False)
    color : `str`, optional
        Colour for the KDE, by default "tab:blue"
    label : `str`, optional
        Label for the plotted KDE, by default None

    Returns
    -------
    ax : `matplotlib Axis`
        Axis on which KDE is plotted
    """

    if seeds is None:
        seeds = np.arange(len(variable))

    # store the KDE values for each bootstrap
    kde_vals = np.zeros((bootstraps, x_count))

    if x_min is None:
        x_min = np.min(variable)
    if x_max is None:
        x_max = np.max(variable)

    # decide on x values to evaluate at (based on log scaling)
    if log_scale[0]:
        print("WARNING: I think this doesn't work", variable)
        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), x_count)
    else:
        x_vals = np.linspace(x_min, x_max, x_count)

    sorted_order = np.argsort(seeds)
    sorted_seeds = seeds[sorted_order]

    # perform bootstrapping
    for i in range(bootstraps):
        _, starts, counts = np.unique(sorted_seeds, return_counts=True, return_index=True)
        res = np.split(sorted_order, starts[1:])
        inds = np.array([np.random.choice(r) if len(r) > 1 else r[0] for r in res])

        print(inds)
        print(inds.shape)

        loop_variable = variable[inds]
        loop_weights = weights[inds] * counts

        # record indices to sample from
        indices = np.arange(len(loop_variable))

        # sample indices
        boot_index = np.random.choice(indices, size=len(indices), replace=True)

        kde = MirroredKDE(loop_variable[boot_index], weights=loop_weights[boot_index],
                          lower_bound=lower_bound, upper_bound=upper_bound, bw_adjust=bw_adjust)
        kde_vals[i] = kde.evaluate(x_vals) * normalisation

    # calculate 1- and 2- sigma percentiles
    percentiles = np.percentile(kde_vals, [15.89, 84.1, 2.27, 97.725], axis=0)

    # plot uncertainties as filled areas
    ax.fill_between(x_vals, percentiles[2], percentiles[3], alpha=0.15, color=color, **kwargs)
    ax.fill_between(x_vals, percentiles[0], percentiles[1], alpha=0.3, color=color, **kwargs)

    # plot the regular kde
    ax.plot(x_vals, np.median(kde_vals, axis=0), color=color, label=label, **kwargs)

    # adjust scales if needed
    if log_scale[0]:
        ax.set_xscale("log")
    if log_scale[1]:
        ax.set_yscale("log")

    return ax


def bootstrapped_kde_fast(variable, weights, ax, seeds=None, bw_adjust=None, normalisation=1,
                     lower_bound=None, upper_bound=None,
                     bootstraps=200, x_min=None, x_max=None, x_count=200, log_scale=(False, False),
                     color="tab:blue", label=None, rng=None, fill_uncertainties=True, **kwargs):
    """Create a bootstrapped weighted KDE plot.

    Bootstrapping is performed by reweighting a single precomputed kernel matrix rather than
    refitting a KDE for each resample, which requires a bandwidth fixed across bootstraps.

    Parameters
    ----------
    variable : `float/array`
        Variable that you want to make a KDE of.
    weights : `float/array`
        Weights associated with each variable (set all to 1 for unweighted)
    ax : `matplotlib Axis`
        Axis on which to plot
    seeds : `int/array`, optional
        Seeds that make the binaries in COMPAS, by default None
    bw_adjust : `float`, optional
        Factor by which to adjust the bandwidth, by default None
    rng : `numpy.random.Generator`, optional
        Random generator to use, by default None

    Returns
    -------
    ax : `matplotlib Axis`
        Axis on which KDE is plotted
    """
    rng = np.random.default_rng(rng)

    variable = np.asarray(variable, dtype=float)
    weights = np.asarray(weights, dtype=float)
    seeds = np.arange(len(variable)) if seeds is None else np.asarray(seeds)

    x_min = variable.min() if x_min is None else x_min
    x_max = variable.max() if x_max is None else x_max
    x_vals = (np.logspace(np.log10(x_min), np.log10(x_max), x_count) if log_scale[0]
              else np.linspace(x_min, x_max, x_count))

    # group rows by seed once, this is invariant across bootstraps
    order = np.argsort(seeds, kind="stable")
    _, starts, counts = np.unique(seeds[order], return_index=True, return_counts=True)
    n_seeds = len(starts)

    # single bandwidth for all bootstraps (Scott's rule on the full weighted sample)
    w_norm = weights / weights.sum()
    n_eff = 1 / np.sum(w_norm**2)
    mean = np.average(variable, weights=weights)
    sigma = np.sqrt(np.average((variable - mean)**2, weights=weights))
    h = sigma * n_eff**(-0.2) * (1.0 if bw_adjust is None else bw_adjust)

    # kernel matrix: contribution of every row to every x value, with reflections at the bounds
    K = np.exp(-0.5 * ((x_vals[None, :] - variable[:, None]) / h)**2)
    if lower_bound is not None:
        K += np.exp(-0.5 * ((x_vals[None, :] - (2 * lower_bound - variable[:, None])) / h)**2)
    if upper_bound is not None:
        K += np.exp(-0.5 * ((x_vals[None, :] - (2 * upper_bound - variable[:, None])) / h)**2)
    K /= h * np.sqrt(2 * np.pi)

    # pick one row per seed, for every bootstrap at once
    pick = order[starts + (rng.random((bootstraps, n_seeds)) * counts).astype(int)]

    # resampling seeds with replacement is a multinomial draw over the seed groups
    multiplicity = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=bootstraps)

    # express both resampling steps as a weight matrix over the original rows
    W = np.zeros((bootstraps, len(variable)))
    np.put_along_axis(W, pick, weights[pick] * counts * multiplicity, axis=1)
    W /= W.sum(axis=1, keepdims=True)

    kde_vals = (W @ K) * normalisation

    # calculate 1- and 2-sigma percentiles
    percentiles = np.percentile(kde_vals, [15.89, 84.1, 2.27, 97.725], axis=0)

    if not isinstance(ax, (list, tuple, np.ndarray)):
        ax = [ax]

    for a in ax:
        if fill_uncertainties:
            a.fill_between(x_vals, percentiles[2], percentiles[3], alpha=0.15, color=color, **kwargs)
            a.fill_between(x_vals, percentiles[0], percentiles[1], alpha=0.3, color=color, **kwargs)
        a.plot(x_vals, np.median(kde_vals, axis=0), color=color, label=label, **kwargs)

        if log_scale[0]:
            a.set_xscale("log")
        if log_scale[1]:
            a.set_yscale("log")

    return ax


def nice_transparent_hist(ax, data, bins, label, colour, density, lw=2, alpha=0.4, cumulative=False, **kwargs):
    ax.hist(data, bins=bins, color=colour, lw=lw, histtype='step', density=density, label=label, cumulative=cumulative, **kwargs)
    ax.hist(data, bins=bins, color=colour, alpha=alpha, density=density, cumulative=cumulative, **kwargs)


def estimate_scale_height_cdf(z, weights=None, R=None, Rlims=(7.5, 8.5), verbose=False):
    """Estimate the scale height of a distribution given z-positions using the cumulative distribution function (CDF).
    This method does not assume a model, but instead just finds the z-value at which the CDF reaches 1 - 1/e ~ 0.63, which corresponds to the scale height for an exponential distribution."""
    z = np.abs(z)
    if R is not None:
        R = R.to(u.kpc).value if hasattr(R, 'unit') else R
        mask = (R >= Rlims[0]) & (R < Rlims[1])
        if verbose:
            print(len(z), "objects before Rlims")
        z = z[mask]
        if verbose:
            print(len(z), "objects in Rlims")

    if hasattr(z, 'unit'):
        z = z.to(u.kpc).value

    if weights is None:
        weights = np.ones_like(z)

    # calculate empirical CDF without any binning
    order = np.argsort(z)
    sorted_z = z[order]
    # cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]

    scale_height = sorted_z[cdf >= (1 - 1 / np.e)][0]

    return scale_height


def bootstrap_cdf(data, n_samples=25_000, n_bootstraps=10, n_bins=250, weights=None, norm=False, fig=None, ax=None,
                  reversed_cdf=True,
                  xlim=(0, np.inf), colour=None, label=None, med_kwargs={}, fill_kwargs={}):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if weights is None:
        weights = np.ones(len(data))

    default_med_kwargs = {'color': colour, 'lw': 3, 'label': label}
    default_fill_kwargs = {'color': colour, 'alpha': 0.15, 'lw': 1}
    med_kwargs_comb = default_med_kwargs.copy()
    med_kwargs_comb.update(med_kwargs)
    fill_kwargs_comb = default_fill_kwargs.copy()
    fill_kwargs_comb.update(fill_kwargs)

    sample_inds = np.random.choice(len(data), size=(n_samples, n_bootstraps), replace=True)
    samples = data[sample_inds]
    sample_weights = weights[sample_inds]
    order = np.argsort(samples, axis=0)

    # apply order to samples and weights to get the CDFs, can't just immediately mask
    # ensure that shapes are the same after resorting!
    ordered_samples = np.take_along_axis(samples, order, axis=0)
    ordered_weights = np.take_along_axis(sample_weights, order, axis=0)

    cdfs = np.cumsum(ordered_weights, axis=0)
    if reversed_cdf:
        cdfs = cdfs.max() - cdfs
    if norm:
        cdfs /= cdfs.max(axis=0)

    # bin the CDFs to make them have a consistent x-axis for plotting
    bin_edges = np.linspace(*xlim, n_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_cdfs = np.zeros((len(bin_centres), n_bootstraps))
    for i in range(n_bootstraps):
        binned_cdfs[:, i] = np.interp(bin_centres, ordered_samples[:, i], cdfs[:, i])

    med = np.median(binned_cdfs, axis=1)
    low = np.percentile(binned_cdfs, 25, axis=1)
    high = np.percentile(binned_cdfs, 75, axis=1)

    ax.fill_between(bin_centres, low, high, **fill_kwargs_comb)
    ax.plot(bin_centres, med, **med_kwargs_comb)
    ax.set_yscale('log')

    return fig, ax, ordered_samples, bin_centres, med, low, high


def four_panel_uncertainties(lisa_sources, lisa_pops, unc_data, wdwd_dist, detectable_pops, counts,
                             n_boot=5000, save=None, show=True):

    height_where_exceeds_wdwds = {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="tight")

    squish = 0.032
    pie_positions = [
        [0, 0.5 - squish, 1/3, 0.5], [1/3, 0.5 - squish, 1/3, 0.5], [2/3, 0.5 - squish, 1/3, 0.5],
        [1/6, squish, 1/3, 0.5], [1/2, squish, 1/3, 0.5],
    ]

    _, _, _, _, wdwd_med, _, _ = bootstrap_cdf(
        np.abs(wdwd_dist[wdwd_dist.tau < 100 * u.Gyr].z.value), n_samples=25_000, n_bootstraps=n_boot, norm=False,
        colour=const.STAR_COLOUR, fig=fig, ax=axes[1, 1], label="Star formation\n(WDWD)", xlim=(0, 10)
    );

    for dco_type in const.DCO_TYPES:
        w = lisa_pops[dco_type].bpp["weights"]

        max_measured_forb = lisa_sources[dco_type].f_orb + unc_data[dco_type]["delta_forb"]
        bootstrap_cdf(np.log10(max_measured_forb.value), n_samples=500, n_bootstraps=n_boot, n_bins=1000,
                    norm=True, colour=const.DCO_COLOURS[dco_type], fig=fig, ax=axes[0, 0], label=dco_type, xlim=(-4.5, -2),
                    reversed_cdf=False)

        min_measured_mass = lisa_sources[dco_type].m_c - unc_data[dco_type]["delta_m_c_over_m_c"] * lisa_sources[dco_type].m_c
        min_measured_mass[min_measured_mass < 0] = min(abs(min_measured_mass))

        bootstrap_cdf(np.log10(min_measured_mass.value), n_samples=500, n_bootstraps=n_boot, n_bins=1000,
                    norm=True, colour=const.DCO_COLOURS[dco_type], fig=fig, ax=axes[0, 1], label=dco_type, xlim=(-1.5, 1.1))


        # pie charts of measurable eccentricity, nested inside the bottom-left slot
        at_least_two_harmonics = unc_data[dco_type]["detectable_harmonics"] >= 2
        ecc_detectable = lisa_sources[dco_type].ecc >= unc.min_detectable_ecc(lisa_sources[dco_type].snr, lisa_sources[dco_type].f_orb * lisa_sources[dco_type].max_snr_harmonic)
        ecc_detectable_no_fmin = lisa_sources[dco_type].ecc >= unc.min_detectable_ecc(lisa_sources[dco_type].snr, lisa_sources[dco_type].f_orb * lisa_sources[dco_type].max_snr_harmonic, f_min=0 * u.mHz)
        w = lisa_pops[dco_type].bpp["weights"]

        measureable_ecc = w[at_least_two_harmonics | ecc_detectable].sum() / w.sum()
        measureable_ecc_no_fmin = w[at_least_two_harmonics | ecc_detectable_no_fmin].sum() / w.sum()

        if measureable_ecc < 0.005:
            measureable_ecc = 0.0
            measureable_ecc_no_fmin = 0.0

        pie_ax = axes[1, 0].inset_axes(pie_positions[const.DCO_TYPES.index(dco_type)], transform=axes[1, 0].transAxes)

        # outer ring is with the frequency cut, inner ring is without
        for vals, rad, alpha in zip(
            [[measureable_ecc, 1 - measureable_ecc], [measureable_ecc_no_fmin, 1 - measureable_ecc_no_fmin]],
            [1, 0.7],
            [1, 0.8]
        ):
            pie_ax.pie(
                vals,
                colors=[const.DCO_COLOURS[dco_type], "lightgrey"],
                startangle=90,
                counterclock=False,
                radius=rad,
                wedgeprops=dict(width=0.3, edgecolor='white', alpha=alpha)
            )
        pie_ax.annotate(dco_type, xy=(0, 0), ha='center', va='center', fontsize=0.6*fs,
                        fontweight='bold', color=const.DCO_COLOURS[dco_type])

        # remove the default ±1.25 padding so the outer ring touches the axis edges
        pie_ax.set_xlim(-1.02, 1.02)
        pie_ax.set_ylim(-1.02, 1.02)

        if counts[dco_type] > 0:
            _, _, _, bin_centres, dco_med, _, _ = bootstrap_cdf(
                np.abs(detectable_pops[dco_type].final_pos[:, 2].value), n_samples=counts[dco_type], n_bootstraps=n_boot,
                norm=False, colour=const.DCO_COLOURS[dco_type], fig=fig, ax=axes[1, 1], xlim=(0, 10)
            );
            above = bin_centres[dco_med > wdwd_med]
            if len(above) > 0:
                first_crossing = above[0]
                height_where_exceeds_wdwds[dco_type] = first_crossing
                print(f"{dco_type} first crosses WDWD at {first_crossing:1.2f} kpc")

                # ax.axvline(first_crossing, 0.7, 1, color=const.DCO_COLOURS[dco_type], ls="--", lw=1.5)
                axes[1, 1].plot([first_crossing, first_crossing], [dco_med[dco_med > wdwd_med][0], 2e3], color=const.DCO_COLOURS[dco_type], ls="--", lw=2)

            else:
                height_where_exceeds_wdwds[dco_type] = np.inf
                print(f"{dco_type} never crosses WDWD")
        else:
            height_where_exceeds_wdwds[dco_type] = np.inf
            print(f"{dco_type} never crosses WDWD (no sources)")

    axes[0, 0].axvline(np.log10(3e-4), color='k', ls="--", lw=2)
    axes[0, 0].axvspan(np.log10(3e-4), -2, color='k', alpha=0.1)
    axes[0, 0].set(
        xlim=(-4.5, -2),
        xlabel=r"$f_{\rm orb} + \Delta f_{\rm orb} \, [\rm Hz]$",
        ylabel=r"$F_{\rm LISA, 8yr} (< f_{\rm orb} + \Delta f_{\rm orb})$",
        yscale="linear",
        ylim=(0, 1),
    )

    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xlabel("Fraction with\nmeasureable eccentricity")

    # axes[1, 1].legend(fontsize=0.65 * fs)
    axes[1, 1].set_xlim(0, 10)
    axes[1, 1].set_ylim(1e-2, 2e3)

    axes[1, 1].xaxis.set_minor_locator(plt.MultipleLocator(0.25))

    top_ax = axes[1, 1].secondary_xaxis('top')
    top_ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    top_ax.set_xticklabels([])

    axes[1, 1].set(
        xlabel="Height above Galactic plane, |z| [kpc]",
        ylabel=r"$N_{\rm LISA, 8yr} (> |z|)$"
    )
        
    axes[0, 1].set(
        yscale="linear",
        ylim=(0, None),
        # ylim=(1e-1, None),
        xlim=(-1.5, 1.1),
        xlabel=r"$\mathcal{M}_c - \Delta \mathcal{M}_c \, [\rm M_\odot]$",
        ylabel=r"$F_{\rm LISA, 8yr} (> \mathcal{M}_c - \Delta \mathcal{M}_c)$"
    )
    max_wdwd_mass = lw.utils.chirp_mass(1.44, 1.44)
    axes[0, 1].axvline(np.log10(max_wdwd_mass), color='k', ls="--", lw=2)
    axes[0, 1].axvspan(-1.5, np.log10(max_wdwd_mass), color='k', alpha=0.1)

    for ax in [axes[0, 0], axes[0, 1]]:
        fake_log_axis(ax, axis="x")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles = [copy(h) for h in handles]
    for h in handles:
        h.set_linewidth(7)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=0.85*fs, bbox_to_anchor=(0.5, 0.99))

    for letter, ax in zip("abcd", axes.flatten()):
        ax.annotate(letter, xy=(0.93 if letter == "b" else 0.04, 0.925 if letter != "c" else 0.09), xycoords="axes fraction", ha='left', va="center_baseline", fontsize=0.9*fs, color="grey", fontweight="bold",
                    bbox=dict(boxstyle="circle", facecolor="lightgrey", edgecolor="grey", linewidth=2))

    axes[1, 1].annotate("Star formation", xy=(0.25, 0.8), xycoords="axes fraction", ha='center', va="center", fontsize=0.7*fs, color=const.STAR_COLOUR, rotation=-42)

    if save:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes, height_where_exceeds_wdwds


def plot_wdwd_distinguishers(
        distinguishers,
        col_labels=[r"$f_{\rm orb}$", r"$\mathcal{M}_c$", r"$e$", r"$|z|$",
                    r"$\{f_{\rm orb}, \mathcal{M}_c\}$", r"$\{f_{\rm orb}, \mathcal{M}_c, e\}$",
                    r"$\{f_{\rm orb}, \mathcal{M}_c, e, |z|\}$"],
        log_scale=True,
        save=None, show=True,
        model_labels=None,
        dots=False
    ):
    models = distinguishers.index.get_level_values("model").unique()
    n_models = distinguishers.index.get_level_values("model").nunique()
    fig, axes = plt.subplots(n_models, 2, figsize=(15, 3 * n_models), gridspec_kw={"width_ratios": [4, 3]})
    
    fig.subplots_adjust(hspace=0.05, wspace=0.02)

    x_vals = np.arange(len(distinguishers.columns))
    x_offset = 0.15

    zeros = []

    for i, col in enumerate(distinguishers.columns):
        for j, dco_type in enumerate(const.DCO_TYPES):
            x_offseted = x_vals[i] + (j - 2) * x_offset

            for model, ax in zip(models, axes[:, 0] if i < 4 else axes[:, 1]):
                val = distinguishers.loc[(model, dco_type), col]
                if val < 0.005:
                    zeros.append((x_offseted, dco_type, ax))
                else:
                    if dots:
                        ax.scatter(x_offseted, val, color=const.DCO_COLOURS[dco_type], s=100, label=dco_type if i == 0 else None)
                    else:
                        ax.bar(x_offseted, val, color=const.DCO_COLOURS[dco_type], width=0.15, label=dco_type if i == 0 else None)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_xticklabels([])
        if not log_scale:
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1)
        else:
            ax.set_ylim(0.0, 1.09)
        ax.grid(which="both", alpha=0.2, axis='y')
        ax.set_axisbelow(True)

    for ax_set, x_set in zip([axes[:, 0], axes[:, 1]], [[0, 1, 2], [4, 5]]):
        for ax in ax_set:
            for x in x_set:
                ax.axvline(x + 0.5, color='k', ls="-", lw=3 if x == 3 else 1)
    
    

    # axes[-1].set_xticks(x_vals)
    for ax in axes[:, 0]:
        ax.set_xlim(-0.5, 3.5)

    for ax, model in zip(axes[:, 1], models):
        ax.set_xlim(3.5, 6.5)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.annotate(model_labels[model] if model_labels is not None else model, xy=(1.05, 0.5), xycoords="axes fraction", ha='center', va="center", fontsize=0.8*fs, rotation=-90)

    if col_labels is None:
        col_labels = distinguishers.columns

    for i, l in zip([0, 1, 2, 3], col_labels):
        axes[-1, 0].annotate(l, xy=(i, -0.05), xycoords=("data", "axes fraction"), ha='center', va="top", fontsize=0.9*fs if i < 4 else 0.7 * fs,
                          rotation=0 if i < 4 else 0)
    for i, l in zip([4, 5, 6], col_labels[4:]):
        axes[-1, 1].annotate(l, xy=(i, -0.05), xycoords=("data", "axes fraction"), ha='center', va="top", fontsize=0.9*fs if i < 4 else 0.7 * fs,
                            rotation=0 if i < 4 else 0)

    axes[3, 0].set_ylabel("Fraction distinguished from WDWDs")

    axes[0, 0].legend(loc='lower center', ncol=len(const.DCO_TYPES), fontsize=0.8*fs, bbox_to_anchor=(0.88, 1.01))

    
    fig.supxlabel("Properties used for distinguishing from WDWDs", fontsize=fs, y=0.07)
        
    if len(zeros) > 0:
        for x, dco_type, ax in zeros:
            og_ymin, og_ymax = ax.get_ylim()
            ax.scatter(x, 1.1e-3 if log_scale else 0.05, color=const.DCO_COLOURS[dco_type], s=100, marker="X", alpha=0.75)
            ax.set_ylim((og_ymin, og_ymax))

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_wdwd_distinguishers_variations(
        distinguishers,
        col_labels=[r"$f_{\rm orb}$", r"$\mathcal{M}_c$", r"$e$", r"$|z|$",
                    r"$\{f_{\rm orb}, \mathcal{M}_c\}$", r"$\{f_{\rm orb}, \mathcal{M}_c, e\}$",
                    r"$\{f_{\rm orb}, \mathcal{M}_c, e, |z|\}$"],
        log_scale=True,
        save=None, show=True,
        model_labels=None,
        model_colours=None,
        cmap="tab10",
        qualitative=True,
        dots=False,
        show_xs=False
    ):
    """Plot the fraction of each DCO type distinguishable from WDWDs, comparing model variations

    Each row is a DCO type and each model variation is a separate bar within each property column.

    Parameters
    ----------
    distinguishers : `pandas.DataFrame`
        Fractions distinguished from WDWDs, indexed by (model, dco_type) with one column per property set
    col_labels : `list` of `str`, optional
        Labels for each column of ``distinguishers``
    log_scale : `bool`, optional
        Whether to use a logarithmic y-axis
    save : `str`, optional
        Path at which to save the figure, by default not saved
    show : `bool`, optional
        Whether to show the figure
    model_labels : `dict`, optional
        Mapping from model name to legend label, by default the model names are used
    model_colours : `dict`, optional
        Mapping from model name to colour, by default drawn from the viridis colourmap
    dots : `bool`, optional
        Whether to plot dots instead of bars

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The figure
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`
        The axes, with shape (n_dco_types, 2)
    """
    models = distinguishers.index.get_level_values("model").unique()
    n_models = len(models)
    n_rows = len(const.DCO_TYPES)

    if model_colours is None:
        if qualitative:
            model_colours = {model: mpl.colormaps[cmap].colors[k] for k, model in enumerate(models)}
        else:
            model_colours = {model: plt.get_cmap(cmap)(k / max(n_models - 1, 1)) for k, model in enumerate(models)}

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3 * n_rows), gridspec_kw={"width_ratios": [4, 3]})
    fig.subplots_adjust(hspace=0.05, wspace=0.03)

    x_vals = np.arange(len(distinguishers.columns))

    # split 80% of each column's width between the models
    bar_width = 0.8 / n_models

    zeros = []

    for i, col in enumerate(distinguishers.columns):
        for k, model in enumerate(models):
            x_offseted = x_vals[i] + (k - (n_models - 1) / 2) * bar_width
            label = (model_labels[model] if model_labels is not None else model) if i == 0 else None

            for dco_type, ax in zip(const.DCO_TYPES, axes[:, 0] if i < 4 else axes[:, 1]):
                val = distinguishers.loc[(model, dco_type), col]
                if val < 0.005:
                    zeros.append((x_offseted, model, ax))
                else:
                    if dots:
                        ax.scatter(x_offseted, val, color=model_colours[model], s=100, label=label)
                    else:
                        ax.bar(x_offseted, val, color=model_colours[model], width=bar_width, label=label)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_xticklabels([])
        if not log_scale:
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1)
        else:
            ax.set_ylim(0.0, 1.09)
        ax.grid(which="both", alpha=0.2, axis='y')
        ax.set_axisbelow(True)

    for ax_set, x_set in zip([axes[:, 0], axes[:, 1]], [[0, 1, 2], [4, 5]]):
        for ax in ax_set:
            for x in x_set:
                ax.axvline(x + 0.5, color='k', ls="-", lw=1)

    for ax in axes[:, 0]:
        ax.set_xlim(-0.5, 3.5)

    # label each row with its DCO type on the right-hand side
    for ax, dco_type in zip(axes[:, 1], const.DCO_TYPES):
        ax.set_xlim(3.5, 6.5)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.annotate(dco_type, xy=(1.05, 0.5), xycoords="axes fraction", ha='center', va="center",
                    fontsize=0.8*fs, rotation=-90, fontweight="bold", color=const.DCO_COLOURS[dco_type])

    if col_labels is None:
        col_labels = distinguishers.columns

    for i, l in zip([0, 1, 2, 3], col_labels):
        axes[-1, 0].annotate(l, xy=(i, -0.05), xycoords=("data", "axes fraction"), ha='center', va="top", fontsize=0.9*fs)
    for i, l in zip([4, 5, 6], col_labels[4:]):
        axes[-1, 1].annotate(l, xy=(i, -0.05), xycoords=("data", "axes fraction"), ha='center', va="top", fontsize=0.7*fs)

    axes[n_rows // 2, 0].set_ylabel("Fraction distinguished from WDWDs")

    axes[0, 0].legend(loc='lower center', ncol=min(n_models, 4), fontsize=0.8*fs, bbox_to_anchor=(0.88, 1.01))

    fig.supxlabel("Properties used for distinguishing from WDWDs", fontsize=fs, y=0.05)

    # mark values that are effectively zero with a cross at the bottom of the axis
    if show_xs:
        for x, model, ax in zeros:
            og_ymin, og_ymax = ax.get_ylim()
            ax.scatter(x, 1.1e-3 if log_scale else 0.05, color=model_colours[model], s=100, marker="X", alpha=0.75)
            ax.set_ylim((og_ymin, og_ymax))

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes