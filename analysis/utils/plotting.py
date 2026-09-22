"""
Utility functions for plotting
"""

import matplotlib
from matplotlib import pyplot as plt
import analysis.utils.binning as bins
import numpy as np


def combine_legends(ax):
    """
    Combine legend entries from multiple axes.

    Parameters
    ----------
    ax: sequence of matplotlib.axes.Axes
        The axes whose legend entries should be combined.

    Returns
    -------
    (list, list):
        Handles and labels of the combined legend.
    """
    legends = [a.get_legend_handles_labels() for a in ax]
    return (l1 + l2 for l1, l2 in zip(*legends))


def plot_legend(ax):
    """
    Plot a standalone legend for the entries plotted in one or multiple axes.

    Parameters
    ----------
    ax: matplotlib.axes.Axes or sequence of matplotlib.axes.Axes
        The axes whose legend entries should be plotted.

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axes
    """
    if isinstance(ax, matplotlib.axes.Axes):
        leg_params = ax.get_legend_handles_labels()
    else:
        leg_params = combine_legends(ax)
    leg_fig, leg_ax = plt.subplots(figsize=(1, 1))
    leg_ax.axis(False)
    leg_fig.set_tight_layout(False)
    leg_fig.legend(*leg_params, loc='center')
    return leg_fig, leg_ax


def plot_binned_values(ax, func, values, binning, selection=None, errors=False, x_errors=True, **plot_args):
    """
    Plot a binned statistic for some values on an existing set of axes.
    The values are divided up into bins of some quantity according to `binning`, with some statistic function applies to
    the values in each bin. The results of the statistic and optionally error bars (if errors are provided by the
    statistic function) for each bin are plotted against the binning quantity on the x-axis. A selection can be provided
    to use only a subset of all the values.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axes to draw the plot.
    func: callable
        A function that takes the values as its first parameter and a boolean for whether to return errors as its second
        parameter and returns the binned results and optional errors.
    values: array_like
        Array of values to be binned and passed to `func`.
    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `analysis.utils.binning.get_binning`.
    selection: indexing expression, optional
        Selection of the values to use in calculating the resolutions (by default use all values).
    errors: bool, optional
        If True, plot error bars calculated as the standard deviation divided by sqrt(N) of the N values in the bin.
    x_errors: bool, optional
        If True, plot horizontal error bars corresponding to the width of the bin, only if `errors` is also True.
    plot_args: optional
        Additional arguments to pass to the plotting function. Note that these may be overridden by arguments
        provided in `runs`.
    """
    plot_args.setdefault('lw', 2)
    binned_values = bins.apply_binning(values, binning, selection)
    x = bins.bin_centres(binning[0])
    if errors:
        y_values, y_errors = func(binned_values, errors)
        x_errors = bins.bin_halfwidths(binning[0]) if x_errors else None
        plot_args.setdefault('marker', '')
        plot_args.setdefault('capsize', 4)
        plot_args.setdefault('capthick', 2)
        ax.errorbar(x, y_values, yerr=y_errors, xerr=x_errors, **plot_args)
    else:
        y = func(binned_values, errors)
        plot_args.setdefault('marker', 'o')
        ax.plot(x, y, **plot_args)
        
# def plot_binned_values_2d(ax, func, values, binning_1, binning_2, selection=None, errors=False, x_errors=True, **plot_args):
    
# #     values # [1,5,3,]
# #     binning_1[0] # bin array [0,2,4,6,10]
    
# #     binning_1[1] # [1,3,1,2,5]
# #     binning_2[1] # [1,2,4,5,1]
    
#     plot_args.setdefault('lw', 2)
# #     binned_values_1 = bins.apply_binning(values, binning_1, selection) # [[2,2,3], [1,3]]
# #     binned_values_2 = bins.apply_binning(values, binning_2, selection)

# #     arr = [[[] for _ in range(binning_1[0].shape[0])] for _ in range(binning_2[0].shape[0])]

# #     for i in range(len(binning_1[0]):
# #         for j in range(len(binning_2[0])):
# #             mask = (binning_1[1] == i) & (binning_2[1] == j)
# #             arr[i][j] = values[mask]

#     arr = [[[] for _ in range(binning_1[0].shape[0])] for _ in range(binning_2[0].shape[0])]

#     for i in range(len(values)):
#         bin_1 = binning_1[i]
#         bin_2 = binning_2[i]
#         arr[bin_1][bin_2].append(values[i])
        
    
#     x = bins.bin_centres(binning_1[0]) 
#     y = bins.bin_centres(binning_2[0]) 

#     vals = func(arr, errors)

#     im = ax.imshow(vals)

# def plot_binned_values_2d(
#     ax,
#     func,
#     values,
#     binning_x,
#     binning_y,
#     selection=None,
#     **plot_args,
# ):
#     binned_values = bins.apply_binning_2d(
#         values,
#         binning_x,
#         binning_y,
#         selection,
#     )

#     z = func(binned_values)

#     im = ax.pcolormesh(
#         binning_x[0],
#         binning_y[0],
#         z,
#         shading="auto",
#         **plot_args,
#     )

#     return im


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
 
 
def _cell_label_colour(cmap, norm, value, label_colour):
    """Choose a readable text colour for a heat map cell, based on the luminance of its fill colour."""
    if label_colour != 'auto':
        return label_colour
    rgba = cmap(float(norm(value)))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return 'w' if luminance < 0.55 else 'k'
 
 
def plot_2d_binned_values(values, binning, counts=None, ax=None, fig_size=None, cmap='viridis', v_lim=None,
                          log_colour=False, centre=None, colorbar=True, colorbar_label='', label_format='{:.2f}',
                          label_size=None, label_colour='auto', show_counts=False, count_format='n={:.0f}',
                          empty_colour='0.9', x_label='', y_label='', title=None, x_scale=None, y_scale=None,
                          **mesh_args):
    """
    Plot a two-dimensional array of binned statistics as a heat map, with the value of each cell drawn on the cell.
 
    Parameters
    ----------
    values: np.ndarray
        Two-dimensional array of shape (n_x_bins, n_y_bins), as returned by `binned_resolutions_2d` etc.
    binning: ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray))
        Two-dimensional binning, as returned by `get_binning_2d`. Only the bin edges are used.
    counts: np.ndarray, optional
        Two-dimensional array of the number of entries in each cell, to annotate cells if `show_counts` is True.
    ax: matplotlib.axes.Axes, optional
        Axes to draw the plot. If not provided, a new figure and axes is created.
    fig_size: (float, float), optional
        Figure size. Ignored if `ax` is provided.
    cmap: str or matplotlib.colors.Colormap, optional
        Colour map for the cells.
    v_lim: (float, float), optional
        Range of the colour scale. By default, the range of the finite values.
    log_colour: bool, optional
        If True, use a logarithmic colour scale (all plotted values must be positive).
    centre: float, optional
        If given, use a diverging colour scale centred on this value, e.g. `centre=0` for bias or difference plots.
        Ignored if `log_colour` is True.
    colorbar: bool, optional
        If True (default), draw a colour bar.
    colorbar_label: str, optional
        Label for the colour bar.
    label_format: str or callable or None, optional
        Format of the value drawn on each cell, either a format string such as '{:.2f}' (default) or a function taking
        the value and returning a string. Use None for no labels.
    label_size: float, optional
        Font size of the cell labels.
    label_colour: str, optional
        Colour of the cell labels. By default ('auto') pick black or white per cell for contrast with the cell colour.
    show_counts: bool, optional
        If True, also draw the number of entries in each cell below its value.
    count_format: str, optional
        Format of the number of entries drawn on each cell.
    empty_colour: str, optional
        Colour used for cells with no value (fewer entries than `min_entries`).
    x_label, y_label: str, optional
        Labels of the axes.
    title: str, optional
        Title of the figure.
    x_scale, y_scale: str, optional
        Scale of the axes, e.g. 'log' when using logarithmically spaced bin edges.
    mesh_args: optional
        Additional arguments passed to `pcolormesh`.
 
    Returns
    -------
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    mesh: matplotlib.collections.QuadMesh
    """
    (x_edges, _), (y_edges, _) = binning
    values = np.asarray(values, dtype=float)
    if values.shape != (x_edges.size - 1, y_edges.size - 1):
        raise ValueError(f"Values of shape {values.shape} do not match the binning "
                         f"({x_edges.size - 1}, {y_edges.size - 1})")
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
 
    finite = values[np.isfinite(values)]
    if v_lim is None:
        v_min, v_max = (np.min(finite), np.max(finite)) if finite.size else (0.0, 1.0)
    else:
        v_min, v_max = v_lim
    if log_colour:
        norm = LogNorm(vmin=max(v_min, np.finfo(float).tiny), vmax=v_max)
    elif centre is not None:
        half_range = max(abs(v_max - centre), abs(centre - v_min)) or 1.0
        norm = TwoSlopeNorm(vcenter=centre, vmin=centre - half_range, vmax=centre + half_range)
    else:
        norm = Normalize(vmin=v_min, vmax=v_max)
 
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(empty_colour)
    ax.set_facecolor(empty_colour)
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values).T, cmap=cmap, norm=norm, **mesh_args)
 
    if label_format is not None:
        formatter = label_format.format if isinstance(label_format, str) else label_format
        x_centres = (x_edges[1:] + x_edges[:-1]) / 2
        y_centres = (y_edges[1:] + y_edges[:-1]) / 2
        for i, x in enumerate(x_centres):
            for j, y in enumerate(y_centres):
                value = values[i, j]
                if not np.isfinite(value):
                    continue
                text = formatter(value)
                if show_counts and counts is not None:
                    text += "\n" + count_format.format(counts[i, j])
                ax.text(x, y, text, ha='center', va='center', fontsize=label_size,
                        color=_cell_label_colour(cmap, norm, value, label_colour))
 
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(mesh, ax=ax, label=colorbar_label)
    return fig, ax, mesh
 
 