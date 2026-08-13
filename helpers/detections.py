import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

from const import DCO_TYPES, DCO_COLOURS

DATA_DIR = "/mnt/ceph/users/twagg/lisa-dcos"

DETECTORS = ["lisa", "decigo"]
POSITIONS = ["initial_pos", "final_pos"]
VARIANTS = ["", "_pessimistic"]

DETECTOR_LABELS = {"lisa": "LISA", "decigo": "DECIGO"}
POSITION_LABELS = {"initial_pos": "Formation", "final_pos": "Present day"}
VARIANT_LABELS = {"": "Optimistic", "_pessimistic": "Pessimistic"}


PERCENTILES = [5, 50, 95]


def load_detections(models, n_targets, detectors=None, dco_types=None, positions=None,
                    variants=None, duration=10, data_dir=DATA_DIR, percentiles=None,
                    warn_missing=False):
    """Load detection percentiles for every combination of model, variant, DCO type,
    detector and position.

    Each file is read exactly once and all requested detector/position columns are
    extracted from it. Combinations whose file, column or normalisation is unavailable are
    filled with NaNs, so the returned frame always contains the full index.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names (also the subdirectory names and `n_targets` index)
    n_targets : `pandas.DataFrame`
        Total number of targets, indexed by model with columns of `{dco_type}{variant}`
    detectors : `list` of `str`, optional
        Detector keys to load, by default `DETECTORS`
    dco_types : `list` of `str`, optional
        DCO types to load, by default `DCO_TYPES`
    positions : `list` of `str`, optional
        Position keys to load, by default `POSITIONS`
    variants : `list` of `str`, optional
        Variant suffixes to load, by default `VARIANTS`
    duration : `int`, optional
        Mission duration in years, by default 10
    data_dir : `str`, optional
        Root directory containing the model subdirectories
    percentiles : `list` of `float`, optional
        Percentiles to record, by default `PERCENTILES`
    warn_missing : `bool`, optional
        Whether to raise a warning summarising the combinations that could not be loaded

    Returns
    -------
    detections : `pandas.DataFrame`
        Numbers of detections with a `(model, variant, dco, detector, position)` MultiIndex
        and columns of `lo`, `mid` and `hi`
    """
    detectors = DETECTORS if detectors is None else detectors
    dco_types = DCO_TYPES if dco_types is None else dco_types
    positions = POSITIONS if positions is None else positions
    variants = VARIANTS if variants is None else variants
    percentiles = PERCENTILES if percentiles is None else percentiles

    # build the full index up front so every requested combination exists, missing or not
    index = pd.MultiIndex.from_product([models, variants, dco_types, detectors, positions],
                                       names=["model", "variant", "dco", "detector", "position"])
    detections = pd.DataFrame(np.nan, index=index, columns=["lo", "mid", "hi"])

    missing = []

    for model in models:
        for variant in variants:
            for dco in dco_types:
                path = os.path.join(data_dir, model, f"{dco}{variant}_f_detect.h5")

                try:
                    f_detect_df = pd.read_hdf(path)
                except (OSError, KeyError):
                    missing.append(f"{model}/{dco}{variant} (file)")
                    continue

                try:
                    norm = n_targets.loc[model][f"{dco}{variant}"]
                except KeyError:
                    missing.append(f"{model}/{dco}{variant} (n_targets)")
                    continue

                for det in detectors:
                    for pos in positions:
                        key = f"{det}_{duration}yr_{pos}"
                        if key not in f_detect_df:
                            missing.append(f"{model}/{dco}{variant} ({key})")
                            continue

                        detections.loc[(model, variant, dco, det, pos)] = np.percentile(
                            f_detect_df[key].values, percentiles) * norm

    if warn_missing and len(missing) > 0:
        warnings.warn(f"filled {len(missing)} combination(s) with NaNs: " + ", ".join(missing))

    return detections


def format_detections(lo, mid, hi, sig=2, missing=r"\nodata"):
    """Format a set of detection percentiles as a LaTeX value with asymmetric uncertainties.

    Parameters
    ----------
    lo, mid, hi : `float`
        Lower, median and upper number of detections
    sig : `int`, optional
        Significant figures to use for the smaller uncertainty, by default 2
    missing : `str`, optional
        Placeholder returned when the values are not finite, by default ``\\nodata``

    Returns
    -------
    value : `str`
        LaTeX-formatted median with +/- uncertainties
    """
    if not np.isfinite(mid):
        return missing

    plus, minus = hi - mid, mid - lo

    # set the precision from the smaller of the two uncertainties
    scale = min(plus, minus)
    if not np.isfinite(scale) or scale <= 0:
        dp = sig
    else:
        dp = max(0, sig - 1 - int(np.floor(np.log10(scale))))

    return rf"${mid:.{dp}f}^{{+{plus:.{dp}f}}}_{{-{minus:.{dp}f}}}$"


def detection_table(models, n_targets, model_labels=None, include_DECIGO=False, dco_types=None,
                    duration=10, data_dir=DATA_DIR, sig=2, missing=r"\nodata",
                    label="tab:detections", caption=None, starred=True, just_tabular=True,
                    detections=None):
    """Write a LaTeX table of detectable DCO numbers, with models as columns.

    Rows are grouped by DCO type, then detector, then the position used for the sky
    localisation. Each model spans a pair of columns for the optimistic and pessimistic
    common-envelope assumptions.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names (also the subdirectory names and `n_targets` index)
    n_targets : `pandas.DataFrame`
        Total number of targets, indexed by model with columns of `{dco_type}{variant}`
    model_labels : `dict`, optional
        Mapping from model name to the label to print, by default the name itself
    include_DECIGO : `bool`, optional
        Whether to include DECIGO rows alongside LISA, by default False. When False the
        detector column is dropped entirely since every row would be labelled identically
    dco_types : `list` of `str`, optional
        DCO types to tabulate, by default `DCO_TYPES`
    duration : `int`, optional
        Mission duration in years, by default 10
    data_dir : `str`, optional
        Root directory containing the model subdirectories
    sig : `int`, optional
        Significant figures for the smaller uncertainty, by default 2
    missing : `str`, optional
        Placeholder for combinations that could not be loaded, by default ``\\nodata``
    label : `str`, optional
        LaTeX label for the table
    caption : `str`, optional
        Table caption
    starred : `bool`, optional
        Whether to use a `table*` environment, by default True
    just_tabular : `bool`, optional
        Whether to return only the `tabular` environment, by default True
    detections : `pandas.DataFrame`, optional
        Pre-loaded output of `load_detections`, loaded here if not supplied

    Returns
    -------
    table : `str`
        The LaTeX table
    """
    model_labels = {} if model_labels is None else model_labels
    dco_types = DCO_TYPES if dco_types is None else dco_types

    detectors = DETECTORS if include_DECIGO else ["lisa"]
    show_detector = len(detectors) > 1

    if detections is None:
        detections = load_detections(models, n_targets, detectors=detectors, dco_types=dco_types,
                                     duration=duration, data_dir=data_dir)

    n_label_cols = 3 if show_detector else 2
    n_data_cols = len(models) * len(VARIANTS)
    n_cols = n_label_cols + n_data_cols
    env = "table*" if starred else "table"

    if not just_tabular:
        lines = [
            rf"\begin{{{env}}}",
            r"\centering",
            rf"\caption{{{caption if caption is not None else ''}}}",
            rf"\label{{{label}}}",
        ]
    else:
        lines = []

    lines.extend([
        r"\begin{tabular}{" + "l" * n_label_cols + "c" * n_data_cols + "}",
        r"\toprule",
    ])

    # first header level: model variation
    row = [""] * n_label_cols
    rules = []
    for i, model in enumerate(models):
        text = model_labels.get(model, model.replace("_", r"\_"))
        row.append(rf"\multicolumn{{{len(VARIANTS)}}}{{c}}{{{text}}}")
        start = n_label_cols + 1 + i * len(VARIANTS)
        rules.append(rf"\cmidrule(lr){{{start}-{start + len(VARIANTS) - 1}}}")
    lines.append(" & ".join(row) + r" \\")
    lines.append(" ".join(rules))

    # second header level: optimistic vs. pessimistic
    row = ["DCO", "Detector", "Position"] if show_detector else ["DCO", "Position"]
    row += [VARIANT_LABELS[variant] for _ in models for variant in VARIANTS]
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    n_dco_rows = len(detectors) * len(POSITIONS)
    for d_ind, dco in enumerate(dco_types):
        for det_ind, det in enumerate(detectors):
            for p_ind, pos in enumerate(POSITIONS):
                # only label the DCO type and detector on the first row of each block
                first = det_ind == 0 and p_ind == 0
                row = [rf"\multirow{{{n_dco_rows}}}{{*}}{{{dco}}}" if first else ""]
                if show_detector:
                    row.append(rf"\multirow{{{len(POSITIONS)}}}{{*}}{{{DETECTOR_LABELS[det]}}}"
                               if p_ind == 0 else "")
                row.append(POSITION_LABELS[pos])

                for model in models:
                    for variant in VARIANTS:
                        lo, mid, hi = detections.loc[(model, variant, dco, det, pos)]
                        row.append(format_detections(lo, mid, hi, sig=sig, missing=missing))

                lines.append(" & ".join(row) + r" \\")

            # light rule between detectors, but not at the end of a DCO block
            if det_ind < len(detectors) - 1:
                lines.append(rf"\cmidrule(lr){{2-{n_cols}}}")

        if d_ind < len(dco_types) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]
    if not just_tabular:
        lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


def plot_detections(models, n_targets, column_labels=None, detectors="lisa", dco_types=None,
                    colours=None, duration=10, data_dir=DATA_DIR, width=0.6, pos_dodge=0.09,
                    log=True, floor=0.1, sharey=False, fig=None, ax=None, show=True,
                    detections=None, save=None, show_sec_ax=False):
    """Plot the number of detectable DCOs for each model variation.

    Each (model, variant) pair gets its own position on the x-axis, DCO types are
    distinguished by colour, and initial/final positions by open/filled markers. Each
    detector gets its own panel, stacked vertically with a shared x-axis. Missing
    combinations are simply absent from the plot.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names (also the subdirectory names and `n_targets` index)
    n_targets : `pandas.DataFrame`
        Total number of targets, indexed by model with columns of `{dco_type}{variant}`
    column_labels : `dict`, optional
        Mapping from `(model, variant)` to the x tick label, by default the fiducial model
        gives "Fiducial"/"Pessimistic" and others append "(pess.)" to the model name
    detectors : `str` or `list` of `str`, optional
        Which detector(s) to plot, one panel each, by default "lisa". Use
        `["lisa", "decigo"]` (or `DETECTORS`) to add a DECIGO panel below
    dco_types : `list` of `str`, optional
        DCO types to plot, by default `DCO_TYPES`
    colours : `dict`, optional
        Mapping from DCO type to colour, by default `DCO_COLOURS`
    duration : `int`, optional
        Mission duration in years, by default 10
    data_dir : `str`, optional
        Root directory containing the model subdirectories
    width : `float`, optional
        Fraction of an x-axis slot across which the DCO types are spread, by default 0.6
    pos_dodge : `float`, optional
        Separation between the initial and final position markers, by default 0.09
    log : `bool`, optional
        Whether to use a logarithmic y-axis, by default True
    floor : `float`, optional
        Lower bounds below this value are truncated to it when `log` is True, by default 0.1
    sharey : `bool`, optional
        Whether the panels share a y-axis, by default False since DECIGO detections
        typically outnumber LISA ones by orders of magnitude
    fig : `matplotlib.figure.Figure`, optional
        Figure on which to plot, created if not supplied
    ax : `matplotlib.axes.Axes` or `list` of `matplotlib.axes.Axes`, optional
        Axis (or one axis per detector) on which to plot, created if not supplied
    show : `bool`, optional
        Whether to show the plot, by default True
    detections : `pandas.DataFrame`, optional
        Pre-loaded output of `load_detections`, loaded here if not supplied. Must cover
        every detector being plotted

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The figure containing the plot
    ax : `matplotlib.axes.Axes` or `numpy.ndarray`
        The axis for a single detector, otherwise an array of axes
    """
    colours = DCO_COLOURS if colours is None else colours
    column_labels = {} if column_labels is None else column_labels
    dco_types = DCO_TYPES if dco_types is None else dco_types
    detectors = [detectors] if isinstance(detectors, str) else list(detectors)

    if detections is None:
        detections = load_detections(models, n_targets, detectors=detectors, dco_types=dco_types,
                                     duration=duration, data_dir=data_dir)

    # each (model, variant) pair becomes its own column on the x-axis
    columns = [(model, variant) for model in models for variant in VARIANTS]

    def default_label(model, variant):
        pretty = model.replace("_", " ").capitalize()
        if model == "fiducial":
            return "Fiducial" if variant == "" else "Pessimistic"
        return pretty if variant == "" else f"{pretty}\n(pess.)"

    labels = [column_labels.get((model, variant), default_label(model, variant))
              for model, variant in columns]

    if fig is None or ax is None:
        fig, ax = plt.subplots(len(detectors), 1, sharex=True, sharey=sharey,
                               figsize=(max(1.6 * len(columns) + 4, 10), 6 * len(detectors)),
                               layout="constrained")
    axes = np.atleast_1d(ax)

    # spread the DCO types evenly within each column, then split initial/final about that
    n_dco = len(dco_types)
    shifts = np.linspace(-width / 2, width / 2, n_dco) if n_dco > 1 else np.zeros(1)

    for axis, detector in zip(axes, detectors):
        for d_ind, dco in enumerate(dco_types):
            for pos in POSITIONS:
                filled = pos == "final_pos"
                offset = shifts[d_ind] + (pos_dodge / 4 if filled else -pos_dodge / 4)

                x, mid, lower, upper = [], [], [], []
                for c_ind, (model, variant) in enumerate(columns):
                    lo, med, hi = detections.loc[(model, variant, dco, detector, pos)]
                    if not np.isfinite(med):
                        continue

                    # truncate error bars that would extend below the axis on a log scale
                    lo = max(lo, floor) if log else lo

                    x.append(c_ind + offset)
                    mid.append(med)
                    lower.append(max(med - lo, 0))
                    upper.append(max(hi - med, 0))

                if len(x) == 0:
                    continue

                axis.errorbar(x, mid, yerr=[lower, upper], fmt="o", markersize=7,
                              color=colours[dco],
                              markerfacecolor=colours[dco] if filled else "none",
                              markeredgecolor=colours[dco], markeredgewidth=1.5, elinewidth=1.5,
                              capsize=3, linestyle="none", zorder=3)


        if show_sec_ax:
            # a constant rescaling, so the inverse is just the reciprocal
            sec_ax = axis.secondary_yaxis(
                "right",
                functions=(lambda y, f=np.sqrt(10/4): y * f,
                        lambda y, f=np.sqrt(10/4): y / f))
            sec_ax.set_ylabel(f"Approximate number of\ndetections (10 yr)")

        # faint dividers between each pair of columns
        for c_ind in range(1, len(columns)):
            axis.axvline(c_ind - 0.5, color="grey", linestyle="dotted", linewidth=2, zorder=0)

        axis.set_xticks(range(len(columns)))
        axis.set_xlim(-0.5, len(columns) - 0.5)
        axis.set_ylabel(f"Number of {DETECTOR_LABELS[detector]}\ndetections ({duration} yr)")
        if log:
            axis.set_yscale("log")

    # only the bottom panel needs tick labels
    axes[-1].set_xticklabels(labels)

    dco_handles = [Line2D([], [], color=colours[dco], marker="o", linestyle="none", label=dco)
                   for dco in dco_types]
    pos_handles = [Line2D([], [], color="grey", marker="o", linestyle="none",
                          label=POSITION_LABELS[pos],
                          markerfacecolor="grey" if pos == "final_pos" else "none",
                          markeredgecolor="grey", markeredgewidth=1.5)
                   for pos in POSITIONS]

    pos_legend = axes[0].legend(handles=pos_handles, loc="lower left", title="Location")
    axes[0].add_artist(pos_legend)

    axes[0].legend(handles=dco_handles, loc="lower center", ncols=len(dco_types),
                   handletextpad=0.0, columnspacing=0.5, bbox_to_anchor=(0.5, 1.02))

    if save is not None:
        plt.savefig(save)

    if show:
        plt.show()

    return fig, ax if len(detectors) > 1 else axes[0]


def n_targets_table(models, n_targets, model_labels=None, dco_types=None, sig=2,
                    missing=r"\nodata", label="tab:n_targets", caption=None, starred=False,
                    just_tabular=True):
    """Write a LaTeX table of the normalisation constants for each DCO type.

    Rows are DCO types and each model spans a pair of columns for the optimistic and
    pessimistic common-envelope assumptions, matching the layout of `detection_table`.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names (also the `n_targets` index)
    n_targets : `pandas.DataFrame`
        Total number of targets, indexed by model with columns of `{dco_type}{variant}`
    model_labels : `dict`, optional
        Mapping from model name to the label to print, by default the name itself
    dco_types : `list` of `str`, optional
        DCO types to tabulate, by default `DCO_TYPES`
    sig : `int`, optional
        Significant figures to show, by default 2
    missing : `str`, optional
        Placeholder for combinations that are absent from `n_targets`, by default ``\\nodata``
    label : `str`, optional
        LaTeX label for the table
    caption : `str`, optional
        Table caption
    starred : `bool`, optional
        Whether to use a `table*` environment, by default False
    just_tabular : `bool`, optional
        Whether to return only the `tabular` environment, by default True

    Returns
    -------
    table : `str`
        The LaTeX table
    """
    model_labels = {} if model_labels is None else model_labels
    dco_types = DCO_TYPES if dco_types is None else dco_types

    n_data_cols = len(models) * len(VARIANTS)
    env = "table*" if starred else "table"

    if not just_tabular:
        lines = [
            rf"\begin{{{env}}}",
            r"\centering",
            rf"\caption{{{caption if caption is not None else ''}}}",
            rf"\label{{{label}}}",
        ]
    else:
        lines = []

    lines.extend([
        r"\begin{tabular}{l" + "c" * n_data_cols + "}",
        r"\hline",
    ])

    # first header level: model variation
    row = [""]
    rules = []
    for i, model in enumerate(models):
        text = model_labels.get(model, model.replace("_", r"\_"))
        row.append(rf"\multicolumn{{{len(VARIANTS)}}}{{c}}{{{text}}}")
        start = 2 + i * len(VARIANTS)
        # rules.append(rf"\cmidrule(lr){{{start}-{start + len(VARIANTS) - 1}}}")
    lines.append(" & ".join(row) + r" \\")
    lines.append(" ".join(rules))

    # second header level: optimistic vs. pessimistic
    row = ["DCO"] + [VARIANT_LABELS[variant] for _ in models for variant in VARIANTS]
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    for dco in dco_types:
        row = [dco]
        for model in models:
            for variant in VARIANTS:
                # fall back to NaN for any model or column that hasn't been run yet
                try:
                    value = n_targets.loc[model][f"{dco}{variant}"]
                except KeyError:
                    value = np.nan
                if value is None or not np.isfinite(value):
                    formatted_val = missing
                else:
                    mantissa, exponent = f"{value:.{sig - 1}e}".split("e")
                    formatted_val = rf"${mantissa} \times 10^{{{int(exponent)}}}$"
                row.append(formatted_val)
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\hline", r"\end{tabular}"]
    if not just_tabular:
        lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)