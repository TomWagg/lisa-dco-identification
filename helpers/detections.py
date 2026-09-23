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
POSITION_LABELS = {"initial_pos": r"$\vec{x}_{\rm i}$", "final_pos": r"$\vec{x}_{\rm f}$"}
VARIANT_LABELS = {"": "Optimistic", "_pessimistic": "Pessimistic"}


PERCENTILES = [5, 50, 95]


def load_detections(models, n_targets, detectors=None, dco_types=None, positions=None,
                    variants=None, duration=4, snr_lim=7, data_dir=DATA_DIR, percentiles=None,
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
        Mission duration in years, by default 4
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
                path_variant = variant if model == "fiducial" else ""
                path = os.path.join(data_dir, model, f"{dco}{path_variant}_f_detect.h5")

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
                        key = f"{det}_{duration}yr_{pos}_SNRgt{snr_lim}"
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
    # if not np.isfinite(scale) or scale <= 0:
    #     dp = sig
    # else:
    #     dp = max(0, sig - 1 - int(np.floor(np.log10(scale))))

    if scale < 10:
        dp = 1
    else:
        dp = 0

    return rf"${mid:.{dp}f}^{{+{plus:.{dp}f}}}_{{-{minus:.{dp}f}}}$"


def detection_table(models, n_targets, model_labels=None, include_DECIGO=False, dco_types=None,
                    duration=8, snr_lim=12, data_dir=DATA_DIR, sig=1, missing=r"\nodata",
                    label="tab:detections", caption=None, starred=True, just_tabular=True,
                    detections=None, fiducial="fiducial", optimistic_label="Optimistic CE", position=None):
    """Write a LaTeX table of detectable DCO numbers, with models as rows.

    Columns are grouped by DCO type, then detector, then the position used for the sky
    localisation. Every model uses the pessimistic common-envelope variant, except that the
    fiducial model is additionally shown with the optimistic variant as though it were a
    separate model, in the row directly after it.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names (also the subdirectory names and `n_targets` index)
    n_targets : `pandas.DataFrame`
        Total number of targets, indexed by model with columns of `{dco_type}{variant}`
    model_labels : `dict`, optional
        Mapping from model name to the label to print, by default the name itself
    include_DECIGO : `bool`, optional
        Whether to include DECIGO columns alongside LISA, by default False. When False the
        detector header level is dropped entirely since every column would be labelled
        identically
    dco_types : `list` of `str`, optional
        DCO types to tabulate, by default `DCO_TYPES`
    duration : `int`, optional
        Mission duration in years, by default 10
    data_dir : `str`, optional
        Root directory containing the model subdirectories
    sig : `int`, optional
        Significant figures for the smaller uncertainty, by default 1
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
    fiducial : `str`, optional
        Name of the fiducial model, the only one also shown with the optimistic variant,
        by default "fiducial"
    optimistic_label : `str`, optional
        Row label for the optimistic fiducial row, by default the optimistic entry of
        `VARIANT_LABELS`
    position : `str`, optional
        Which entry of `POSITIONS` to tabulate, by default None, which includes every
        position. When a single position is given the position header level is dropped
        since every column would be labelled identically

    Returns
    -------
    table : `str`
        The LaTeX table
    """
    model_labels = {} if model_labels is None else model_labels
    dco_types = DCO_TYPES if dco_types is None else dco_types

    if position is not None and position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, not '{position}'")
    positions = POSITIONS if position is None else [position]
    show_position = len(positions) > 1

    optimistic, pessimistic = VARIANTS[0], VARIANTS[1]
    optimistic_label = VARIANT_LABELS[optimistic] if optimistic_label is None else optimistic_label

    # each row is a (model, variant, label) triple, with the optimistic fiducial
    # inserted directly after the fiducial model
    table_rows = []
    for model in models:
        text = model_labels.get(model, model.replace("_", r"\_"))
        table_rows.append((model, pessimistic, text))
        if model == fiducial:
            table_rows.append((model, optimistic, optimistic_label))

    # rows are labelled by letter (to be described in the caption) to keep the table narrow
    letters = [chr(ord("A") + i) for i in range(len(table_rows))]
    key = {letter: text for letter, (_, _, text) in zip(letters, table_rows)}

    detectors = DETECTORS if include_DECIGO else ["lisa"]
    show_detector = len(detectors) > 1

    if detections is None:
        detections = load_detections(models, n_targets, detectors=detectors, dco_types=dco_types,
                                     duration=duration, data_dir=data_dir, snr_lim=snr_lim)

    # every data column is a (dco, detector, position) combination, in this order
    data_cols = [(dco, det, pos) for dco in dco_types for det in detectors for pos in positions]
    n_per_det = len(positions)
    n_per_dco = len(detectors) * n_per_det
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
        r"\begin{tabular}{l" + "c" * len(data_cols) + "}",
        r"\toprule",
    ])

    # header levels as (group labels, columns spanned by each group), skipping any level
    # whose labels would be identical across every column
    levels = [(list(dco_types), n_per_dco)]
    if show_detector:
        levels.append(([DETECTOR_LABELS[det] for _ in dco_types for det in detectors], n_per_det))
    if show_position:
        levels.append(([POSITION_LABELS[pos] for _, _, pos in data_cols], 1))

    for l_ind, (texts, span) in enumerate(levels):
        # the model column heading sits on the bottom level, which needs no rules beneath
        last = l_ind == len(levels) - 1
        row, rules = ["Model" if last else ""], []
        for g_ind, text in enumerate(texts):
            start = 2 + g_ind * span
            row.append(rf"\multicolumn{{{span}}}{{c}}{{{text}}}" if span > 1 else text)
            if not last:
                rules.append(rf"\cmidrule(lr){{{start}-{start + span - 1}}}")
        lines.append(" & ".join(row) + r" \\")
        if rules:
            lines.append(" ".join(rules))
    lines.append(r"\midrule")

    # one row per model (plus the optimistic fiducial)
    for letter, (model, variant, _) in zip(letters, table_rows):
        row = [key[letter]]
        for dco, det, pos in data_cols:
            lo, mid, hi = detections.loc[(model, variant, dco, det, pos)]
            row.append(format_detections(lo, mid, hi, sig=sig, missing=missing))
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    if not just_tabular:
        lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


def plot_detections(models, n_targets, column_labels=None, detectors="lisa", dco_types=None,
                    colours=None, duration=4, snr_lim=7, data_dir=DATA_DIR, width=0.6, pos_dodge=0.09,
                    log=True, floor=0.1, sharey=False, fig=None, ax=None, show=True,
                    detections=None, save=None, show_sec_ax=False, sec_duration=8,
                    fiducial="fiducial", optimistic_label=None, letter_labels=False,
                    ylim=None):
    """Plot the number of detectable DCOs for each model variation.

    Each model gets its own position on the x-axis, using the pessimistic common-envelope
    variant, except that the fiducial model is additionally shown with the optimistic variant
    as though it were a separate model, directly after it (matching `detection_table`). DCO
    types are distinguished by colour, and initial/final positions by open/filled markers.
    Each detector gets its own panel, stacked vertically with a shared x-axis. Missing
    combinations are simply absent from the plot.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names (also the subdirectory names and `n_targets` index)
    n_targets : `pandas.DataFrame`
        Total number of targets, indexed by model with columns of `{dco_type}{variant}`
    column_labels : `dict`, optional
        Mapping from `(model, variant)` to the x tick label, by default "Fiducial" for the
        fiducial model, `optimistic_label` for its optimistic variant, and the prettified
        model name otherwise. Ignored if `letter_labels` is True
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
    save : `str`, optional
        Path at which to save the figure, not saved if not supplied
    show_sec_ax : `bool`, optional
        Whether to add a secondary y-axis showing the approximate number of detections for a
        mission lasting `sec_duration` years, by default False
    sec_duration : `float`, optional
        Mission duration in years for the secondary y-axis, by default 8. Detections are
        rescaled by `sqrt(sec_duration / duration)`
    fiducial : `str`, optional
        Name of the fiducial model, the only one also shown with the optimistic variant,
        by default "fiducial"
    optimistic_label : `str`, optional
        Tick label for the optimistic fiducial column, by default "Optimistic CE"
    letter_labels : `bool`, optional
        Whether to label columns A, B, C, ... to match the tables, by default False

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
    optimistic_label = "Optimistic CE" if optimistic_label is None else optimistic_label

    if detections is None:
        detections = load_detections(models, n_targets, detectors=detectors, dco_types=dco_types,
                                     duration=duration, data_dir=data_dir, snr_lim=snr_lim)

    # one column per model (pessimistic), plus the optimistic fiducial straight after it
    rows, _ = _model_rows(models, fiducial=fiducial)
    columns = [(model, variant) for _, model, variant in rows]

    def default_label(model, variant):
        if model == fiducial:
            return "Fiducial" if variant == VARIANTS[1] else optimistic_label
        return model.replace("_", " ").capitalize()

    if letter_labels:
        labels = [letter for letter, _, _ in rows]
    else:
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

    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ticks = [1, 4, 10, 40, 100, 400]

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

                    if ylim is not None and lo < ylim[0]:
                        ax.scatter(c_ind + offset, 0.1, color=colours[dco], marker="v", facecolor="none" if not filled else colours[dco])

                    # truncate error bars that would extend below the axis on a log scale
                    lo = max(lo, floor) if log else lo

                    x.append(c_ind + offset)
                    mid.append(med)
                    lower.append(max(med - lo, 0))
                    upper.append(max(hi - med, 0))

                    # if the fiducial, pessimistic, final positions, shade across every axis
                    if model == "fiducial" and variant == "_pessimistic" and pos == "final_pos":
                        # axis.axhspan(mid[0] - lower[0], mid[0] + upper[0], color=colours[dco], alpha=0.1, zorder=0)
                        axis.axhline(mid[0], color=colours[dco], zorder=0, linestyle=":", lw=1)

                if len(x) == 0:
                    continue

                axis.errorbar(x, mid, yerr=[lower, upper], fmt="o", markersize=7,
                              color=colours[dco],
                              markerfacecolor=colours[dco] if filled else "none",
                              markeredgecolor=colours[dco], markeredgewidth=1.5, elinewidth=1.5,
                              capsize=3, linestyle="none", zorder=3)

        # dividers between each column
        for c_ind in range(1, len(columns)):
            axis.axvline(c_ind - 0.5, color="k", linestyle="-", linewidth=1, zorder=0)

        axis.set_xticks(range(len(columns)))
        axis.set_xlim(-0.5, len(columns) - 0.5)
        axis.set_ylabel(f"Number of {DETECTOR_LABELS[detector]} detections\n({duration} yr observations, S/N > {snr_lim})")
        if log:
            axis.set_yscale("log")

        # faint horizontal lines at 1, 10, 100
        # for line in ticks:
        #     if np.log10(line) % 1 == 0:
        #         axis.axhline(line, color="grey", linestyle="dotted", linewidth=1, zorder=0)

        if log:
            axis.set_yticks(ticks)
            axis.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # only the bottom panel needs tick labels
    axes[-1].set_xticklabels(labels, fontsize=0.8*fs)

    dco_handles = [Line2D([], [], color=colours[dco], marker="o", linestyle="none", label=dco)
                   for dco in dco_types]
    POSITION_LABELS = {
        "initial_pos": r"Formation",
        "final_pos": r"Present day"
    }
    pos_handles = [Line2D([], [], color="grey", marker="o", linestyle="none",
                          label=POSITION_LABELS[pos],
                          markerfacecolor="grey" if pos == "final_pos" else "none",
                          markeredgecolor="grey", markeredgewidth=1.5)
                   for pos in POSITIONS]

    pos_legend = axes[0].legend(handles=pos_handles, loc="lower center", title="Location",
                                bbox_to_anchor=(0.22, 0.015),
                                framealpha=1.0)
    pos_legend = axes[0].legend(handles=pos_handles, loc="lower center", ncol=2,
                                bbox_to_anchor=(0.2, 1.02), handletextpad=0.0, columnspacing=0.5,
                                framealpha=1.0)
    axes[0].add_artist(pos_legend)

    axes[0].legend(handles=dco_handles, loc="lower center", ncols=len(dco_types),
                   handletextpad=0.0, columnspacing=0.5, bbox_to_anchor=(0.7, 1.02))

    if show_sec_ax:
        # a constant rescaling (detections scale as sqrt of duration), so the inverse is just
        # division by the same factor
        factor = np.sqrt(sec_duration / duration)
        for axis in axes:
            sec_ax = axis.secondary_yaxis("right",
                                          functions=(lambda y, f=factor: y * f,
                                                     lambda y, f=factor: y / f))
            sec_ax.set_ylabel(f"Approximate number of\ndetections ({sec_duration:g} yr)")
            sec_ax.set_yticks(ticks)
            sec_ax.set_yticklabels(ticks)

    if ylim is not None:
        for axis in axes:
            axis.set_ylim(ylim)

    if save is not None:
        plt.savefig(save)

    if show:
        plt.show()

    return fig, ax if len(detectors) > 1 else axes[0]


def _model_rows(models, model_labels=None, fiducial="fiducial", optimistic_label=None):
    """Build the lettered rows shared by `detection_table` and `n_targets_table`.

    Every model uses the pessimistic common-envelope variant, except that the fiducial model
    is additionally included with the optimistic variant as though it were a separate model,
    directly after it.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names
    model_labels : `dict`, optional
        Mapping from model name to its full label, by default the name itself
    fiducial : `str`, optional
        Name of the fiducial model, by default "fiducial"
    optimistic_label : `str`, optional
        Full label for the optimistic fiducial row, by default the optimistic entry of
        `VARIANT_LABELS`

    Returns
    -------
    rows : `list` of `tuple`
        One (letter, model, variant) triple per row
    key : `dict`
        Mapping from each row letter to its full model label
    """
    model_labels = {} if model_labels is None else model_labels
    optimistic, pessimistic = VARIANTS[0], VARIANTS[1]
    optimistic_label = VARIANT_LABELS[optimistic] if optimistic_label is None else optimistic_label

    entries = []
    for model in models:
        entries.append((model, pessimistic, model_labels.get(model, model.replace("_", r"\_"))))
        if model == fiducial:
            entries.append((model, optimistic, optimistic_label))

    # rows are labelled by letter (to be described in the caption) to keep tables narrow
    rows, key = [], {}
    for i, (model, variant, text) in enumerate(entries):
        letter = chr(ord("A") + i)
        rows.append((letter, model, variant))
        key[letter] = text
    return rows, key

def n_targets_table(models, n_targets, model_labels=None, dco_types=None, sig=2,
                    missing=r"\nodata", label="tab:n_targets", caption=None, starred=False,
                    just_tabular=True, fiducial="fiducial", optimistic_label="Optimistic CE",
                    return_key=False, norms={"BHBH": 1e5, "BHNS": 1e5, "NSNS": 1e4, "BHWD": 1e4, "NSWD": 1e5}):
    """Write a LaTeX table of the normalisation constants for each DCO type.

    Rows are models, labelled by letter, and columns are DCO types, matching the layout of
    `detection_table`. Every model uses the pessimistic common-envelope variant, except that
    the fiducial model is additionally shown with the optimistic variant as though it were a
    separate model, in the row directly after it.

    Parameters
    ----------
    models : `list` of `str`
        Model variation names (also the `n_targets` index)
    n_targets : `pandas.DataFrame`
        Total number of targets, indexed by model with columns of `{dco_type}{variant}`
    model_labels : `dict`, optional
        Mapping from model name to its full label, used only in the key returned when
        `return_key` is True, by default the name itself
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
    fiducial : `str`, optional
        Name of the fiducial model, the only one also shown with the optimistic variant,
        by default "fiducial"
    optimistic_label : `str`, optional
        Full label for the optimistic fiducial row, used only in the key, by default the
        optimistic entry of `VARIANT_LABELS`
    return_key : `bool`, optional
        Whether to also return the mapping from row letter to full model label, by default
        False

    Returns
    -------
    table : `str`
        The LaTeX table
    key : `dict`
        Mapping from each row letter to its full model label, only returned if `return_key`
        is True
    """
    dco_types = DCO_TYPES if dco_types is None else dco_types
    rows, key = _model_rows(models, model_labels=model_labels, fiducial=fiducial,
                            optimistic_label=optimistic_label)
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
        r"\begin{tabular}{l" + "c" * len(dco_types) + "}",
        r"\hline",
    ])

    # single header row: model letter then one column per DCO type
    lines.append(" & ".join(["Model"] + list(dco_types)) + r" \\")
    lines.append(r" & [$10^5$] & [$10^5$] & [$10^4$] & [$10^4$] & [$10^5$] \\")
    lines.append(r"\midrule")

    # one row per model (plus the optimistic fiducial)
    for letter, model, variant in rows:
        row = [key[letter]]
        # if model == fiducial and variant == VARIANTS[0]:
        #     row[0] = optimistic_label
        for dco in dco_types:
            # fall back to NaN for any model or column that hasn't been run yet
            try:
                value = n_targets.loc[model][f"{dco}{variant}"]
            except KeyError:
                value = np.nan
            if value is None or not np.isfinite(value):
                formatted_val = missing
            else:
                normed_val = value / norms.get(dco, 1)
                formatted_val = f"{normed_val:.2f}" if value > 100 else f"{normed_val:.3f}"
                # mantissa, exponent = f"{value:.{sig - 1}e}".split("e")
                # formatted_val = rf"${mantissa} \times 10^{{{int(exponent)}}}$"
            row.append(formatted_val)
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\hline", r"\end{tabular}"]
    if not just_tabular:
        lines.append(rf"\end{{{env}}}")
    table = "\n".join(lines)
    return (table, key) if return_key else table