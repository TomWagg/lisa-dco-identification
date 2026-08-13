def nice_transparent_hist(ax, data, bins, label, colour, density, lw=2, alpha=0.4, cumulative=False, **kwargs):
    ax.hist(data, bins=bins, color=colour, lw=lw, histtype='step', density=density, label=label, cumulative=cumulative, **kwargs)
    ax.hist(data, bins=bins, color=colour, alpha=alpha, density=density, cumulative=cumulative, **kwargs)