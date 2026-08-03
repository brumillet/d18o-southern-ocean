# -*- coding: utf-8 -*-
"""
Basin masks and water-mass criteria, defined once.

The notebooks wrote these boundaries out inline, each time with the column
names of whatever DataFrame was at hand, so the same basin was defined slightly
differently in different cells. The numbers below are the ones used in the
evaluation notebook.

Longitudes are on 0-360 throughout, consistent with the products.
"""

import numpy as np

# Northern limit of what this study calls the Southern Ocean.
SO_LAT_MAX = -40.0

# Neutral density and absolute salinity window used to isolate the waters
# influenced by the North Atlantic contribution in the deep Southern Ocean.
NA_GAMMA_MIN, NA_GAMMA_MAX = 27.9, 28.05
NA_SALINITY_MIN = 34.8

# Wider neutral density window used when splitting the Southern Ocean
# compilation into its individual datasets.
SO_GAMMA_MIN, SO_GAMMA_MAX = 27.0, 28.6


def basin_masks(df, lat_col='Latitude', lon_col='Longitude'):
    """
    Boolean masks for the basins used in the RMSE comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        Must hold latitude and longitude columns, longitude on 0-360.
    lat_col, lon_col : str, optional
        Column names.

    Returns
    -------
    dict of str to pandas.Series
        ``'Global'``, ``'Southern Ocean'``, ``'Atlantic Ocean'``,
        ``'Indian Ocean'`` and ``'Pacific Ocean'``.

    Notes
    -----
    The three low-latitude basins stop at 40 S, where the Southern Ocean mask
    takes over, so the five masks do not overlap but do not tile the ocean
    either (the Arctic and the marginal seas fall outside all but 'Global').
    """
    lat, lon = df[lat_col], df[lon_col]
    return {
        'Global': lat <= 100,
        'Southern Ocean': lat <= SO_LAT_MAX,
        'Atlantic Ocean': (lat >= SO_LAT_MAX) & ((lon >= 295) | (lon <= 30)),
        'Indian Ocean': (lat >= SO_LAT_MAX) & (lat <= 20) & (lon >= 30) & (lon <= 120),
        'Pacific Ocean': (lat >= SO_LAT_MAX) & (lat <= 70) & (lon >= 120) & (lon <= 295),
    }


def na_influenced(df, gamma_col='gamma', salinity_col='absolute_salinity'):
    """
    Mask of the deep waters carrying a North Atlantic signature.

    Selected as ``27.9 <= gamma_n <= 28.05`` together with an absolute salinity
    at or above 34.8 psu.

    Parameters
    ----------
    df : pandas.DataFrame
    gamma_col, salinity_col : str, optional
        Column names for neutral density and absolute salinity.

    Returns
    -------
    pandas.Series of bool
    """
    return ((df[gamma_col] >= NA_GAMMA_MIN) & (df[gamma_col] <= NA_GAMMA_MAX)
            & (df[salinity_col] >= NA_SALINITY_MIN))


def southern_ocean(df, lat_col='Latitude'):
    """Mask of the observations south of :data:`SO_LAT_MAX`."""
    return df[lat_col] <= SO_LAT_MAX


def error_depth_levels(shallow_levels, deep_start=1000, deep_step=100, n_deep=50):
    """
    Depth levels at which the cumulative RMSE against the compilation is
    evaluated.

    The evaluation notebook uses the product's own levels down to about 1000 m,
    then a regular 100 m spacing below, so that the deep ocean is sampled evenly
    regardless of the product's vertical grid.

    Parameters
    ----------
    shallow_levels : array-like
        The product levels to use above `deep_start`.
    deep_start, deep_step, n_deep : optional
        First depth, spacing and count of the regular part.

    Returns
    -------
    list of float
    """
    levels = list(np.asarray(shallow_levels, dtype=float))
    levels += [deep_start + i * deep_step for i in range(n_deep)]
    return levels
