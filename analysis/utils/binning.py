"""
Utility functions for binning events in some quantity and manipulating other quantities based on the binning
"""
import numpy as np
from watchmal.utils.math import binomial_error


def get_binning(x, bins=None, minimum=None, maximum=None, width=None):
    """
    Finds the indices of the bins to which each value in input array belongs, for a set of bins specified either as an
    array of bin edges, number of bins or bin width

    Parameters
    ----------
    x: array_like
        Input array to be binned.
    bins: array_like, optional
        If `bins` is an int, it defines the number of equal-width bins in the range (200, by default). If `bins` is an
        array, it is the array of bin edges and must be 1-dimensional and monotonic.
    minimum: int or real, optional
        Lowest bin lower edge (by default use minimum value in `x`). Not used if `bins` is an ndarray of bin edges.
    maximum: int or real, optional
        Highest bin upper edge (by default use minimum value in `x`). Not used if `bins` is an ndarray of bin edges.
    width: int or real, optional
        Width of bins to generate equal width bins if `bins` is None

    Returns
    -------
    bins: np.ndarray
        array of bin edges
    indices: np.ndarray
        output array of indices, of same shape as x
    """
    bin_array = np.array(bins)
    if bin_array.size == 1:
        if minimum is None:
            minimum = np.min(x)
        if maximum is None:
            maximum = np.max(x)
        if bins is None:
            bin_array = np.arange(minimum, maximum+width, width)
        else:
            bin_array = np.linspace(minimum, maximum, bins+1)
    indices = np.digitize(x, bin_array)
    return bin_array, indices


def apply_binning(values, binning, selection=...):
    """
    This function bins values according to the indices returned by `get_binning`. Returns a list of arrays where the nth
    array contains the values assigned to the nth bin.

    Parameters
    ----------
    values: array_like
        Values to be partitioned into bins

    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `get_binning`

    selection: index expression, optional
        If provided, then `values` is indexed using this selection so that the values that pass the selection are binned
        and returned

    Returns
    -------
    list of np.ndarray
        List of arrays of values assigned to each bin
    """
    if len(values) // len(selection) == 2:
        selection = np.concatenate((selection, selection))
    data = values[selection]
    data_bins = binning[1][selection]
    return [data[data_bins == b] for b in range(1, binning[0].size)]
# def apply_binning_2d(values, binning_x, binning_y, selection=None):
#     """
#     Bin values in two quantities.

#     Returns
#     -------
#     list[list[np.ndarray]]
#         Shape is [n_y_bins][n_x_bins].
#     """
#     values = np.asarray(values)

#     if selection is None:
#         selection = np.ones(len(values), dtype=bool)

#     values = values[selection]
#     x_indices = binning_x[1][selection]
#     y_indices = binning_y[1][selection]

#     n_x = len(binning_x[0]) - 1
#     n_y = len(binning_y[0]) - 1

#     return [
#         [
#             values[(x_indices == x + 1) & (y_indices == y + 1)]
#             for x in range(n_x)
#         ]
#         for y in range(n_y)
#     ]

def unapply_binning(binned_values, binning, selection=...):
    """
    Reverses the effect of `apply_binning`. Takes a list of arrays corresponding to values assigned to bins and returns
    a single array of values ordered as they were before they were binned.

    Parameters
    ----------
    binned_values: list of np.ndarray
        Binned values returned by `apply_binning`

    binning: (np.ndarray, np.ndarray)
        Array of bin edges and array of bin indices, returned from `get_binning`

    selection: index expression, optional
        If provided, the returned array will match the original length before the selection of `apply_binning` was
        applied. The missing entries that do not exist in `binned_values` due to not passing the selection are filled
        with zeros.

    Returns
    -------
    list of np.ndarray
        List of arrays of values assigned to each bin
    """
    data = np.zeros(binning[1].shape)
    data_bins = binning[1][selection]
    for b, v in enumerate(binned_values):
        data[selection][data_bins == b+1] = v


def binned_resolutions(binned_residuals, return_errors=True):
    """
    Calculate resolution defined as 68th percentile of the absolute residuals for each of a list of arrays of residuals

    Parameters
    ----------
    binned_residuals: list of array_like
        list of arrays of float residuals in each bin
    return_errors: bool
        if True, return array of standard error on the mean of each list of values (as a proxy for the standard error on
        each list's 68th percentile).

    Returns
    -------
    resolutions: np.ndarray
        array of resolutions of the bins' residuals
    errors: np.ndarray
        array of standard errors on the means of the residuals
    """
    resolutions = np.array([np.nanquantile(np.abs(x), 0.68) for x in binned_residuals])
    if return_errors:
        errors = binned_std_errors(binned_residuals)
        return resolutions, errors
    else:
        return resolutions

# def binned_resolutions_2d(arr, return_errors=True):
#     resolutions = np.array([[np.nanquantile(np.abs(arr[x,y]), 0.68) for x in arr.shape[0] for y in arr.shape[1]])
    
#     return resolutions

def binned_resolutions_2d(arr, return_errors=False):
    return np.array([
        [
            np.nanquantile(np.abs(cell), 0.68)
            if len(cell) > 0
            else np.nan
            for cell in row
        ]
        for row in arr
    ])

def binned_quantiles(binned_values, quantile):
    """
    Calculate quantiles of the values for each of a list of arrays of values

    Parameters
    ----------
    binned_values: list of array_like
        list of arrays of float values in each bin
    quantile: float
        quantile value to find in each bin

    Returns
    -------
    np.ndarray
        array of quantiles of the bins' values
    """
    return np.array([np.nanquantile(x, quantile) for x in binned_values])


def binned_mean(binned_values, return_errors=True):
    """
    Calculate mean of the values for each of a list of arrays of values

    Parameters
    ----------
    binned_values: list of array_like
        list of arrays of values in each bin
    return_errors: bool
        if True, return array of standard error on the mean of each list of values

    Returns
    -------
    means: np.ndarray
        array of means of the bins' values
    errors: np.ndarray, optional
        array of standard errors of the means
    """
    means = np.array([np.mean(x) for x in binned_values])
    if return_errors:
        errors = binned_std_errors(binned_values)
        return means, errors
    else:
        return means


def binned_efficiencies(binned_cut, return_errors=True, reverse=False):
    """
    Calculate percentage of true values (and binomial errors) in each arrays of booleans in a list

    Parameters
    ----------
    binned_cut: list of array_like
        list of arrays of booleans in each bin
    reverse: bool
        If True, reverse the cut to give percentage of events failing the cut. By default, give percentage of events
        passing the cut
    return_errors: bool
        if True, return array of each list of booleans' binomial standard error

    Returns
    -------
    efficiencies: np.ndarray
        array of percentage of true values in each bin
    errors: np.ndarray, optional
        array of binomial standard errors
    """
    efficiencies = binned_mean(binned_cut, return_errors=False)*100
    if reverse:
        efficiencies = 100 - efficiencies
    if return_errors:
        errors = binned_binomial_errors(binned_cut) * 100
        return efficiencies, errors
    else:
        return efficiencies


def binned_std_errors(binned_residuals):
    """
    Calculate standard errors for each of a list of arrays of residuals

    Parameters
    ----------
    binned_residuals: list of array_like
        list of arrays of float residuals in each bin

    Returns
    -------
    np.ndarray
        array of standard errors of the bins' residuals
    """
    return np.array([np.std(x)/np.sqrt(x.size) for x in binned_residuals])


def binned_binomial_errors(binned_results):
    """
    Calculate standard binomial errors for each of a list of arrays of binomial trial results

    Parameters
    ----------
    binned_results: list of array_like
        list of arrays of boolean binomial results in each bin

    Returns
    -------
    np.ndarray
        array of binomial errors of the bins' results
    """
    return np.array([binomial_error(x) for x in binned_results])


def bin_centres(bins):
    """Array of bin centres for an array of bin edges"""
    return (bins[1:]+bins[:-1])/2


def bin_halfwidths(bins):
    """Array of bin half-widths for an array of bin edges"""
    return (bins[1:]-bins[:-1])/2

 
def get_binning_2d(x, y, x_bins=None, y_bins=None, x_range=(None, None), y_range=(None, None),
                   x_width=None, y_width=None):
    """
    Bin events in two quantities at once, by applying `get_binning` to each of them.
 
    Parameters
    ----------
    x: array_like
        Input array to be binned along the x-axis.
    y: array_like
        Input array to be binned along the y-axis.
    x_bins, y_bins: array_like, optional
        Number of equal-width bins, or array of bin edges, for each axis. See `get_binning`.
    x_range, y_range: (int or real, int or real), optional
        Lowest lower edge and highest upper edge of each axis. See `get_binning`.
    x_width, y_width: int or real, optional
        Width of equal-width bins for each axis, if the number of bins is not given. See `get_binning`.
 
    Returns
    -------
    ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray))
        Pair of (bin edges, bin indices) tuples, as returned by `get_binning`, for the x and y axes respectively.
    """
    x_binning = get_binning(x, x_bins, x_range[0], x_range[1], x_width)
    y_binning = get_binning(y, y_bins, y_range[0], y_range[1], y_width)
    if x_binning[1].shape != y_binning[1].shape:
        raise ValueError("The x and y binning quantities must have the same shape (one entry per event)")
    return x_binning, y_binning
 
    
def apply_binning_2d(values, binning, selection=...):
    """
    Two-dimensional equivalent of `apply_binning`. Returns a nested list of arrays, where element [i][j] contains the
    values assigned to the ith bin in x and the jth bin in y.
 
    As in `apply_binning`, values falling outside the binning range (including values equal to the highest bin edge,
    which `np.digitize` assigns to the overflow bin) are not included in any bin.
 
    Parameters
    ----------
    values: array_like
        Values to be partitioned into bins.
    binning: ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray))
        Two-dimensional binning, as returned by `get_binning_2d`.
    selection: index expression, optional
        If provided, then `values` is indexed using this selection so that only the values that pass the selection are
        binned. As in `apply_binning`, if there are twice as many values as entries in the selection (e.g. two rings per
        event), the selection is duplicated to cover both.
 
    Returns
    -------
    list of list of np.ndarray
        Nested list of arrays of values assigned to each cell, of shape (n_x_bins, n_y_bins).
    """
    (x_edges, x_indices), (y_edges, y_indices) = binning
    values = np.asarray(values)
    if selection is not ... and len(values) // len(selection) == 2:
        selection = np.concatenate((selection, selection))
    data = values[selection]
    x_bins = x_indices[selection]
    y_bins = y_indices[selection]
    n_x, n_y = x_edges.size - 1, y_edges.size - 1
 
    in_range = (x_bins >= 1) & (x_bins <= n_x) & (y_bins >= 1) & (y_bins <= n_y)
    flat_bins = (x_bins[in_range] - 1) * n_y + (y_bins[in_range] - 1)
    data = data[in_range]
    order = np.argsort(flat_bins, kind="stable")
    flat_bins, data = flat_bins[order], data[order]
    cell_edges = np.searchsorted(flat_bins, np.arange(n_x * n_y + 1))
    return [[data[cell_edges[i * n_y + j]:cell_edges[i * n_y + j + 1]] for j in range(n_y)] for i in range(n_x)]

    
def binned_resolutions_2d(binned_residuals, min_entries=1):
    """
    Calculate resolution defined as the 68th percentile of the absolute residuals for each cell of a two-dimensional
    binning. Cells with too few entries are returned as NaN.
 
    Parameters
    ----------
    binned_residuals: list of list of array_like
        Nested list of arrays of float residuals in each cell, returned from `apply_binning_2d`.
    min_entries: int, optional
        Cells containing fewer than this many entries are returned as NaN, to avoid quoting a resolution from a handful
        of events. Default is 1, i.e. only empty cells are NaN.
 
    Returns
    -------
    np.ndarray
        Two-dimensional array of resolutions of the cells' residuals, of shape (n_x_bins, n_y_bins).
    """
    return np.array([[np.nanquantile(np.abs(cell), 0.68) if np.size(cell) >= max(min_entries, 1) else np.nan
                      for cell in row] for row in binned_residuals])
 
def binned_mean_2d(binned_values, min_entries=1):
    """
    Calculate the mean of the values in each cell of a two-dimensional binning, for use as a bias.
 
    Parameters
    ----------
    binned_values: list of list of array_like
        Nested list of arrays of values in each cell, returned from `apply_binning_2d`.
    min_entries: int, optional
        Cells containing fewer than this many entries are returned as NaN.
 
    Returns
    -------
    np.ndarray
        Two-dimensional array of means of the cells' values.
    """
    return np.array([[np.mean(cell) if np.size(cell) >= max(min_entries, 1) else np.nan
                      for cell in row] for row in binned_values])
 
def binned_quantiles_2d(binned_values, quantile, min_entries=1):
    """
    Calculate a quantile of the values in each cell of a two-dimensional binning.
 
    Parameters
    ----------
    binned_values: list of list of array_like
        Nested list of arrays of values in each cell, returned from `apply_binning_2d`.
    quantile: float
        Quantile value to find in each cell.
    min_entries: int, optional
        Cells containing fewer than this many entries are returned as NaN.
 
    Returns
    -------
    np.ndarray
        Two-dimensional array of quantiles of the cells' values.
    """
    return np.array([[np.nanquantile(cell, quantile) if np.size(cell) >= max(min_entries, 1) else np.nan
                      for cell in row] for row in binned_values])
 
def binned_counts_2d(binned_values):
    """
    Count the entries in each cell of a two-dimensional binning.
 
    Parameters
    ----------
    binned_values: list of list of array_like
        Nested list of arrays of values in each cell, returned from `apply_binning_2d`.
 
    Returns
    -------
    np.ndarray
        Two-dimensional array of the number of entries in each cell.
    """
    return np.array([[np.size(cell) for cell in row] for row in binned_values])
 
def binned_std_errors_2d(binned_residuals):
    """
    Calculate standard errors for each cell of a two-dimensional binning.
 
    Parameters
    ----------
    binned_residuals: list of list of array_like
        Nested list of arrays of float residuals in each cell, returned from `apply_binning_2d`.
 
    Returns
    -------
    np.ndarray
        Two-dimensional array of standard errors of the cells' residuals.
    """
    return np.array([[np.std(cell) / np.sqrt(np.size(cell)) if np.size(cell) else np.nan
                      for cell in row] for row in binned_residuals])
 