# -*- coding: utf-8 -*-
"""
Loading and cleaning of the delta-18O observation sets.

Each compilation arrives in its own format with its own missing-value
conventions, and the notebooks repeated that cleaning at every use, which is
how the same dataset ended up loaded three times with slightly different
filters. The rules are collected here:

============  =========================  ============================================
dataset       file                       cleaning
============  =========================  ============================================
GISS          giss_d18o.txt              longitude to 0-360; ``'**'`` and ``-999``
                                         are missing values; ``d18O`` is text
GISS (h5)     giss_d18o_modified.h5      as above, plus gamma_n / absolute salinity
Aoki KY2018   Datasheet_KY2018_*.csv     ``;`` separated, units row skipped,
                                         ``-999`` missing
CISE-LOCEAN   SI-Wisotopes-V3.csv        no header, quality flag ``iO <= 2``
Bostock       d18o_helen_bostock.xlsx    rows without a longitude are dropped
============  =========================  ============================================

Every loader returns a DataFrame with the original column names, so existing
notebook code keeps working unchanged.
"""

import numpy as np
import pandas as pd

import config


# The column names of the CISE-LOCEAN file, which ships without a header row.
LOCEAN_COLUMNS = [
    'Cruise name', 'station id', 'bottle number', 'day', 'month', 'year', 'hour',
    'minute', 'latitude', 'longitude', 'pressure (db)', 'temperature (°C)', 'it',
    'salinity (pss-78)', 'is', 'dissolved oxygen (micromol/kg)', 'io2', 'd18O', 'iO',
    'dD', 'iD', 'd-excess', 'id', 'method type',
]


def to_0_360(longitudes):
    """
    Wrap longitudes onto 0-360, the convention used by every product here.

    Parameters
    ----------
    longitudes : array-like

    Returns
    -------
    numpy.ndarray
    """
    return np.asarray(longitudes, dtype=float) % 360.0


def load_giss(which='global', data_path=None):
    """
    Load the GISS delta-18O compilation.

    Longitudes are wrapped to 0-360, the ``-999`` depth sentinel becomes NaN and
    the ``'**'`` delta-18O sentinel is dropped so that the column can be cast to
    float.

    Parameters
    ----------
    which : {'global', 'indian', 'modified'}, optional
        ``'global'`` reads ``giss_d18o.txt``, ``'indian'`` the Indian Ocean
        extract, ``'modified'`` the HDF5 version that also carries ``gamma_n``
        and ``absolute_salinity``.
    data_path : str, optional
        Override the directory; defaults to :data:`config.DATA_PATH`.

    Returns
    -------
    pandas.DataFrame
        With a numeric ``d18O`` column.
    """
    if data_path is not None:
        config.set_data_path(data_path)

    if which == 'global':
        df = pd.read_table(config.data_file('giss'))
    elif which == 'indian':
        df = pd.read_table(config.data_file('giss_indian'))
    elif which == 'modified':
        df = pd.read_hdf(config.data_file('giss_modified'))
    else:
        raise ValueError(f"which must be 'global', 'indian' or 'modified', got {which!r}")

    df = df.copy()
    df['Longitude'] = to_0_360(df['Longitude'])
    df.loc[df['Depth'] == -999.0, 'Depth'] = np.nan
    # d18O is read as text because of the '**' sentinel used for missing values.
    df = df[df['d18O'] != '**'].copy()
    df['d18O'] = df['d18O'].astype(float)
    return df


def load_aoki(min_pressure=200.0, data_path=None):
    """
    Load the Aoki KY2018 Southern Ocean dataset.

    Parameters
    ----------
    min_pressure : float, optional
        Drop samples shallower than this pressure in dbar (default 200, the
        value used in the evaluation notebook to stay below the mixed layer).
    data_path : str, optional
        Override the directory.

    Returns
    -------
    pandas.DataFrame
    """
    if data_path is not None:
        config.set_data_path(data_path)

    # Row 1 holds the units, not data.
    df = pd.read_csv(config.data_file('aoki'), sep=';', skiprows=[1])
    df = df.where((df['dO18'] != -999.) & (df['CTDPRS_DBAR'] >= min_pressure))
    return df.dropna(ignore_index=True)


def load_locean(max_flag=2, min_pressure=None, data_path=None):
    """
    Load the CISE-LOCEAN water isotope database.

    Parameters
    ----------
    max_flag : int, optional
        Keep only samples whose delta-18O quality flag ``iO`` is at most this
        (default 2).
    min_pressure : float or None, optional
        If given, keep only samples at or below this pressure in dbar.
    data_path : str, optional
        Override the directory.

    Returns
    -------
    pandas.DataFrame
    """
    if data_path is not None:
        config.set_data_path(data_path)

    df = pd.read_csv(config.data_file('locean'), delimiter=';', names=LOCEAN_COLUMNS)
    df = df.where(df['iO'] <= max_flag)
    if min_pressure is not None:
        df = df.where(df['pressure (db)'] >= min_pressure)
    return df.dropna(how='all')


def load_bostock(data_path=None):
    """
    Load the Helen Bostock Southern Ocean dataset.

    Rows without a longitude are dropped, as in the original notebook.

    Parameters
    ----------
    data_path : str, optional
        Override the directory.

    Returns
    -------
    pandas.DataFrame
    """
    if data_path is not None:
        config.set_data_path(data_path)

    df = pd.read_excel(config.data_file('bostock'))
    return df.dropna(subset=['Longitude [°]']).reset_index()


def add_product_columns(df, products, coords, columns=None):
    """
    Interpolate every product at the observation positions, in one call.

    This is the step the notebooks repeated for each dataset, one line per
    product, with the column names spelled out by hand every time.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations. Modified in place and also returned.
    products : dict of str to products.Product
        As returned by :func:`products.load_products`.
    coords : tuple of str
        Names of the ``(depth, latitude, longitude)`` columns of `df`. Depth may
        be a pressure in dbar, as the notebooks do throughout.
    columns : dict of str to str, optional
        Product key to output column name. Defaults to ``'d18O_' + key``. Pass
        an explicit mapping to keep an existing notebook's column names.

    Returns
    -------
    pandas.DataFrame
        `df`, with one new column per product.

    Notes
    -----
    Longitudes must already be on 0-360; use :func:`to_0_360` if not. Points
    outside a product's grid, or on one of its land cells, come back as NaN, so
    a subsequent ``dropna()`` over the whole frame keeps only the observations
    that *every* product covers.
    """
    depth_col, lat_col, lon_col = coords
    for key, product in products.items():
        name = (columns or {}).get(key, 'd18O_' + key)
        df[name] = product.sample(df[depth_col], df[lat_col], df[lon_col])
    return df


def add_climatology_columns(df, interpolators, coords, columns=None):
    """
    Interpolate climatology fields (neutral density, salinity, ...) at the
    observation positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations. Modified in place and also returned.
    interpolators : dict of str to callable
        As returned by :func:`products.load_climatology`.
    coords : tuple of str
        Names of the ``(depth, latitude, longitude)`` columns of `df`.
    columns : dict of str to str, optional
        Climatology variable name to output column name. Defaults to the
        variable name itself.

    Returns
    -------
    pandas.DataFrame
    """
    depth_col, lat_col, lon_col = coords
    for name, interp in interpolators.items():
        out = (columns or {}).get(name, name)
        df[out] = interp((df[depth_col], df[lat_col], df[lon_col]))
    return df
