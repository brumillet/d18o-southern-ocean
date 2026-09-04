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
D18O_LABEL = r'$\delta^{18}O$ (‰)'
D18O_DIFF_LABEL = r'$\Delta \delta^{18}O$ (‰)'
GAMMA_LABEL = r'$\gamma_n$ (kg m$^{-3}$)'

# Cumulative-RMSE-against-depth axis, written f(d) in the notebook text.
ERROR_DEPTH_LABEL = r'$f(d)$ (‰)'

# Grey used for the neutral-density lines drawn across the profile panels, so
# that they read as guides rather than as another data series.
DENSITY_LINE_COLOR = '0.45'


def panel_label(ax, index, text='', x=0.02, y=0.03, color='black', font=14, gap=0.06,
                letter_color='black', letter_weight='bold'):
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
        Colour of `text`.
    font : int, optional
        Font size.
    gap : float, optional
        Horizontal offset of `text` from the letter, in axes coordinates.
    letter_color, letter_weight : optional
        Colour and weight of the letter itself. The panel letters of the
        multi-panel figures are black and bold; the dataset tags of the profile
        grids are coloured like their dataset and not bold, hence the
        parameters.

    Returns
    -------
    None
    """
    letter = uf.panels_letter_parenthesis[index] if isinstance(index, int) else index
    ax.text(x, 1 - y, letter, transform=ax.transAxes, fontsize=font,
            fontweight=letter_weight, color=letter_color, va='top', ha='left')
    if text:
        ax.text(x + gap, 1 - y, text, transform=ax.transAxes, fontsize=font,
                color=color, va='top', ha='left')


def salinity_max_near(df, gamma, salinity_col='absolute_salinity',
                      gamma_col='gamma_n', half_width=0.05):
    """
    Highest bottle salinity found around a neutral-density surface.

    The profile panels draw a horizontal line at the neutral density where each
    dataset's delta-18O turns over; this is the salinity of the water sitting on
    that surface, annotated just under the line.

    Parameters
    ----------
    df : pandas.DataFrame
        One dataset's bottles.
    gamma : float
        Neutral density of the line.
    salinity_col, gamma_col : str, optional
        Column names. The compilations disagree: ``gamma_n`` in the HDF5 GISS
        version and in the optimizer's datasets, ``Gamma`` once interpolated
        from the climatology.
    half_width : float, optional
        Half-width of the density window the maximum is taken over.

    Returns
    -------
    float
        NaN if the dataset carries no finite salinity at all. When the window is
        empty the maximum over the whole dataset is returned instead, so a line
        drawn outside the sampled density range still gets a number.
    """
    salinity = np.asarray(df[salinity_col], dtype=float)
    density = np.asarray(df[gamma_col], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        window = salinity[(density >= gamma - half_width) & (density <= gamma + half_width)]
        value = np.nanmax(window) if window.size else np.nan
        if not np.isfinite(value) and salinity.size:
            value = np.nanmax(salinity)
    return value


def density_line(ax, gamma, salinity=None, x=0.98, font=13,
                 color=DENSITY_LINE_COLOR, ls='--', alpha=0.8, gap=11):
    """
    Draw the neutral-density line of a profile panel, labelled.

    The density value goes above the line and, when given, the salinity maximum
    of the bottles around it goes just below.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    gamma : float
        Neutral density at which to draw the line.
    salinity : float or None, optional
        Salinity maximum to annotate; see :func:`salinity_max_near`. Nothing is
        written when this is None or NaN.
    x : float, optional
        Right edge of both labels, in axes coordinates. They are right-aligned
        so that the longer salinity line cannot run off a narrow panel.
    font, color, ls, alpha : optional
        Appearance of the line and its labels.
    gap : float, optional
        Vertical offset of the labels from the line, in points.

    Returns
    -------
    None

    Notes
    -----
    The labels are offset in *points* rather than in density units because the
    panels use the ``custom_scale`` of :mod:`custom_density_scale`
    (``log1p(28.6 - gamma)``), on which a fixed offset in gamma covers a
    different distance depending on where the line sits.
    """
    ax.axhline(y=gamma, ls=ls, color=color, alpha=alpha)

    def annotate(text, dy):
        # A faint box keeps the labels readable where they cross the scatter.
        ax.annotate(text, xy=(x, gamma), xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(0, dy),
                    fontsize=font, color=color, ha='right',
                    va='bottom' if dy > 0 else 'top',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.0))

    annotate(r'$\gamma_n$ = ' + format(gamma, '.2f'), gap)
    if salinity is not None and np.isfinite(salinity):
        annotate(r'$S_{max}$ = ' + format(salinity, '.2f'), -gap)


def southern_map(ax, extent=(-180, 180, -90, -20), font=14, resolution='110m',
                 xlocs=None, ylocs=None, draw_labels=None):
    """
    Dress a cartopy axes the way the map panels of the notebooks do.

    Coastlines, a grey land mask and labelled gridlines, then the extent, all
    with the same sizes so that the maps of different figures match.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes created with ``projection=ccrs.PlateCarree()``.
    extent : tuple, optional
        ``(lon_min, lon_max, lat_min, lat_max)``.
    font : int, optional
        Font size of the figure; the gridline labels are one point smaller.
    resolution : str, optional
        Coastline resolution, as accepted by ``ax.coastlines``.
    xlocs, ylocs : array-like, optional
        Gridline positions; matplotlib picks them when omitted.
    draw_labels : dict, optional
        Passed to ``ax.gridlines``; defaults to the bottom / left / right
        labelling used throughout.

    Returns
    -------
    cartopy.mpl.gridliner.Gridliner
    """
    # Imported here so that this module stays importable without cartopy.
    import cartopy.feature as cfeature

    ax.coastlines(resolution=resolution)
    ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=0)

    kwargs = {'draw_labels': draw_labels or {'bottom': 'x', 'right': 'y', 'left': 'y'}}
    if xlocs is not None:
        kwargs['xlocs'] = list(xlocs)
    if ylocs is not None:
        kwargs['ylocs'] = list(ylocs)
    gl = ax.gridlines(**kwargs)
    gl.xlabel_style = {'size': font - 1}
    gl.ylabel_style = {'size': font - 1}

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return gl


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
