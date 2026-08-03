# -*- coding: utf-8 -*-
"""
Plotting helpers shared by the notebooks.

:mod:`usefull_functions` already holds the axis and colorbar helpers; this
module adds the pieces the evaluation notebook repeated inline: the panel
label idiom, the Southern Ocean averages, and the RMSE-against-depth
computation.

Product colours and labels live in :mod:`products` so that they stay next to
the products themselves.
"""

import warnings

import numpy as np

import usefull_functions as uf

# Axis label for delta-18O, spelled out here so every figure agrees.
D18O_LABEL = r'$\delta^{18}O_{sw}$ (‰)'
D18O_DIFF_LABEL = r'$\Delta \delta^{18}O_{sw}$ (‰)'
GAMMA_LABEL = r'$\gamma_n$'


def panel_label(ax, index, text='', x=0.02, y=0.03, color='black', font=14, gap=0.06):
    """
    Put a bold ``(a)``-style letter, and optionally a title, inside a panel.

    The notebooks did this with two ``ax.text`` calls in data coordinates, which
    had to be re-tuned whenever the axis limits changed. Using axes coordinates
    means the label sits in the same place whatever the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    index : int or str
        Panel number (0 -> ``'(a)'``) or a ready-made string.
    text : str, optional
        Short title placed after the letter.
    x, y : float, optional
        Position in axes coordinates, measured from the top left.
    color : str, optional
        Colour of `text` (the letter is always black).
    font : int, optional
        Font size.
    gap : float, optional
        Horizontal offset of `text` from the letter, in axes coordinates.

    Returns
    -------
    None
    """
    letter = uf.panels_letter_parenthesis[index] if isinstance(index, int) else index
    ax.text(x, 1 - y, letter, transform=ax.transAxes, fontsize=font,
            fontweight='bold', va='top', ha='left')
    if text:
        ax.text(x + gap, 1 - y, text, transform=ax.transAxes, fontsize=font,
                color=color, va='top', ha='left')


def zonal_mean(field3d):
    """
    Circumpolar mean of a ``(depth, lat, lon)`` field.

    Parameters
    ----------
    field3d : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Shape ``(depth, lat)``. Latitude bands that are entirely land come back
        as NaN rather than raising.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(field3d, axis=2)


def area_mean(field3d, lat):
    """
    cos(lat)-weighted horizontal mean of a ``(depth, lat, lon)`` field, level by
    level.

    Weighting matters here: on a regular latitude-longitude grid the polar rows
    hold far less ocean than the equatorial ones, and an unweighted mean over
    the Southern Ocean gives them the same say.

    Parameters
    ----------
    field3d : numpy.ndarray
        Shape ``(depth, lat, lon)``. NaNs are excluded from both the sum and the
        weight total.
    lat : array-like
        Latitude axis matching the second dimension.

    Returns
    -------
    numpy.ndarray
        One value per level, NaN where the level has no valid cell.
    """
    weights = np.broadcast_to(
        np.cos(np.deg2rad(np.asarray(lat, dtype=float)))[None, :, None], np.shape(field3d))
    finite = np.isfinite(field3d)
    w = np.where(finite, weights, 0.0)
    total = np.sum(w, axis=(1, 2))
    numerator = np.sum(np.where(finite, field3d, 0.0) * w, axis=(1, 2))
    return np.where(total > 0, numerator / np.where(total > 0, total, 1.0), np.nan)


def cumulative_rmse(df, obs_column, product_columns, depth_levels, depth_column='Depth'):
    """
    RMSE of each product against the observations, over all samples below a
    series of depths.

    This is the measure the evaluation notebook plots against depth to find the
    level below which the reconstructions beat the LeGrande & Schmidt product:

    .. math:: f(d) = \\sqrt{\\langle (\\delta^{18}O_{obs} - \\delta^{18}O_p)^2
                            \\rangle_{d_i \\geq d}}

    Parameters
    ----------
    df : pandas.DataFrame
        Observations, already restricted to the region of interest.
    obs_column : str
        Column holding the measured delta-18O.
    product_columns : sequence of str
        Columns holding the interpolated products.
    depth_levels : sequence of float
        Depths at which to evaluate.
    depth_column : str, optional
        Column holding the sample depth.

    Returns
    -------
    tuple
        ``(rmse, n_points)`` where ``rmse`` has shape
        ``(len(depth_levels), len(product_columns))`` and ``n_points`` gives the
        number of samples entering each depth level.

    Notes
    -----
    The RMSE at a given level is computed on whatever samples `df` still holds
    below it, so the curves are only comparable between products if `df` has
    already been reduced to the rows where every product is defined.
    """
    rmse = np.full((len(depth_levels), len(product_columns)), np.nan)
    n_points = np.zeros(len(depth_levels))

    obs = df[obs_column].values.astype(float)
    depth = df[depth_column].values.astype(float)
    products = {c: df[c].values.astype(float) for c in product_columns}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for i_depth, level in enumerate(depth_levels):
            below = depth >= level
            n_points[i_depth] = below.sum()
            for i_col, column in enumerate(product_columns):
                rmse[i_depth, i_col] = np.sqrt(
                    np.nanmean((obs[below] - products[column][below]) ** 2))
    return rmse, n_points
