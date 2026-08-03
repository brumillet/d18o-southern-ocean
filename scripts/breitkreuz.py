# -*- coding: utf-8 -*-
"""
Loader for the Breitkreuz et al. (2018) gridded delta-18O of seawater product.

Reference
---------
Breitkreuz, C., Paul, A., Kurahashi-Nakamura, T., Losch, M., Schulz, M. (2018):
A dynamical reconstruction of the global monthly-mean oxygen isotopic
composition of seawater. Journal of Geophysical Research: Oceans, 123(10),
7206-7219. https://doi.org/10.1029/2018JC014300

Conventions in the raw file that this module normalises
------------------------------------------------------
* longitude runs -179.5 .. 179.5   -> rolled to 0.5 .. 359.5 to match the rest
  of this project (OCIM / TMI / NEMO / GISS all use 0-360)
* depth is negative downwards (-25 .. -4855 m) -> flipped to positive downwards
* land / below-bathymetry points are NaN
* lat/lon are stored as 2D arrays even though the 1 deg grid is fully regular;
  they are collapsed back to 1D vectors here
"""

import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


# Plotting identity of the product, kept here so every figure stays consistent.
BREITKREUZ_LABEL = 'B18 product'
BREITKREUZ_COLOR = '#9467bd'          # purple, distinct from NEMO/OCIM/TMI/LS
BREITKREUZ_FIELD = 'd18O_B18'         # column name used in the observation frames

# Depth of the deepest and shallowest cell centre in the raw file. Used by the
# padding logic and worth having on hand when interpreting NaNs.
BREITKREUZ_DEPTH_MIN = 25.0
BREITKREUZ_DEPTH_MAX = 4855.0


def load_breitkreuz(nc_path, month=None, pad_surface=True, pad_cyclic=True,
                    depth_bottom=None, variables=('D18O', 'SALT', 'THETA')):
    """
    Load the Breitkreuz et al. (2018) product on its regular 1 deg lat-lon grid.

    Parameters
    ----------
    nc_path : str
        Path to ``D18O_Breitkreuz_et_al_2018.nc``.
    month : int or None, optional
        1-12 to extract a single calendar month, or None (default) for the
        annual mean. 
    pad_surface : bool, optional
        If True (default), repeat the shallowest level at depth 0.
    pad_cyclic : bool, optional
        If True (default), wrap one column around each side in longitude
        (359.5 -> -0.5 and 0.5 -> 360.5) so that points falling between the
        last and first grid column are interpolated rather than returned as
        NaN.
    depth_bottom : float or None, optional
        If a depth is given, repeat the deepest level down to it, so that
        samples below the last cell centre (4855 m) get the bottom value
        instead of NaN. ``None`` (default) adds no bottom level, leaving
        everything below 4855 m undefined.
    variables : tuple of str, optional
        Which raw fields to load. ``'D18O'`` -> ``d18o``, ``'SALT'`` ->
        ``salinity``, ``'THETA'`` -> ``theta``.

    Returns
    -------
    xarray.Dataset
        Dimensions ``(depth, lat, lon)`` with ``depth`` positive downwards and
        ``lon`` in 0-360, both strictly increasing.

    Notes
    -----
    The salinity in this file is the model salinity (practical salinity), not
    the absolute salinity used elsewhere in this project. Convert with
    ``gsw.SA_from_SP`` before comparing to ``dsClim['absolute_salinity']``.
    """
    name_map = {'D18O': 'd18o', 'SALT': 'salinity', 'THETA': 'theta'}
    unknown = set(variables) - set(name_map)
    if unknown:
        raise ValueError(f'Unknown Breitkreuz variables {sorted(unknown)}; '
                         f'expected a subset of {sorted(name_map)}')

    raw = xr.open_dataset(nc_path)
    try:
        # The 1 deg grid is regular, so the 2D coordinate arrays collapse to 1D.
        lat = raw['lat_1deg_center'].values[:, 0].astype(float)
        lon = raw['lon_1deg_center'].values[0, :].astype(float)
        depth = -raw['depth_center'].values.astype(float)   # -> positive downwards

        fields = {}
        for var in variables:
            values = raw[var + '_1deg'].values.astype(float)   # (month, depth, y, x)
            if month is None:
                # Land columns are NaN in all 12 months; nanmean returns NaN for
                # them but warns, which is noise rather than information here.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    values = np.nanmean(values, axis=0)
            else:
                if not 1 <= month <= 12:
                    raise ValueError(f'month must be in 1..12, got {month}')
                values = values[month - 1]
            fields[name_map[var]] = values                     # (depth, y, x)
    finally:
        raw.close()

    # --- longitude: -179.5..179.5 -> 0.5..359.5 -----------------------------
    lon = lon % 360.0
    order = np.argsort(lon)
    lon = lon[order]
    fields = {k: v[:, :, order] for k, v in fields.items()}

    # --- depth: file is ordered shallow -> deep once sign is flipped --------
    if not np.all(np.diff(depth) > 0):
        d_order = np.argsort(depth)
        depth = depth[d_order]
        fields = {k: v[d_order] for k, v in fields.items()}

    # The two directions are independent: holding the 25 m value up to the
    # surface is reasonable, extending the 4855 m value into the abyss is not,
    # so the latter only happens when a depth is asked for explicitly.
    if pad_surface:
        depth = np.concatenate(([0.0], depth))
        fields = {k: np.concatenate((v[:1], v), axis=0) for k, v in fields.items()}

    if depth_bottom is not None:
        if depth_bottom <= depth[-1]:
            raise ValueError(f'depth_bottom ({depth_bottom}) must be deeper than '
                             f'the last model level ({depth[-1]})')
        depth = np.concatenate((depth, [depth_bottom]))
        fields = {k: np.concatenate((v, v[-1:]), axis=0) for k, v in fields.items()}

    if pad_cyclic:
        lon = np.concatenate((lon[-1:] - 360.0, lon, lon[:1] + 360.0))
        fields = {k: np.concatenate((v[:, :, -1:], v, v[:, :, :1]), axis=2)
                  for k, v in fields.items()}

    ds = xr.Dataset(
        {k: (('depth', 'lat', 'lon'), v) for k, v in fields.items()},
        coords={'depth': depth, 'lat': lat, 'lon': lon},
    )
    ds['d18o'].attrs = {'units': 'permil VSMOW', 'long_name': 'seawater d18O'}
    if 'salinity' in ds:
        ds['salinity'].attrs = {'units': 'psu', 'long_name': 'practical salinity'}
    if 'theta' in ds:
        ds['theta'].attrs = {'units': 'degC', 'long_name': 'potential temperature'}
    ds.attrs = {
        'source': 'Breitkreuz et al. (2018), PANGAEA 10.1594/PANGAEA.889922',
        'grid': '1 deg lat-lon (interpolated from the native cubed-sphere grid)',
        'month': 'annual mean' if month is None else f'month {month}',
        'pad_surface': str(pad_surface),
        'pad_cyclic': str(pad_cyclic),
        'depth_bottom': 'none' if depth_bottom is None else str(depth_bottom),
    }
    return ds


def breitkreuz_interpolator(ds, variable='d18o', fill_value=np.nan):
    """
    Build a ``(depth, lat, lon)`` interpolator over a loaded Breitkreuz field.

    Matches the calling convention of the other product interpolators in this
    project, so it can be dropped in next to ``interpolator_d18o_rcst_ocim``
    and friends::

        interp((df['Depth'], df['Latitude'], df['Longitude']))

    Longitudes must be given in 0-360, depth positive downwards.

    Parameters
    ----------
    ds : xarray.Dataset
        Output of :func:`load_breitkreuz`.
    variable : str, optional
        Field to interpolate (default ``'d18o'``).
    fill_value : float or None, optional
        Value returned outside the grid. ``None`` enables extrapolation; the
        default NaN matches the other products in the evaluation notebook.

    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
    """
    return RegularGridInterpolator(
        (ds['depth'].values, ds['lat'].values, ds['lon'].values),
        ds[variable].values,
        method='linear', bounds_error=False, fill_value=fill_value,
    )


def regrid_to(ds, l_depth, l_lat, l_lon, variable='d18o', fill_value=np.nan):
    """
    Sample a Breitkreuz field onto another product's regular grid.

    Used by the direct product-to-product comparison figures, where the
    reconstructions and B18 have to live on a common grid before differencing.

    Parameters
    ----------
    ds : xarray.Dataset
        Output of :func:`load_breitkreuz`.
    l_depth, l_lat, l_lon : array-like
        Target 1D axes (depth positive downwards, longitude in 0-360).
    variable : str, optional
        Field to sample (default ``'d18o'``).
    fill_value : float or None, optional
        Value outside the source grid.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(l_depth), len(l_lat), len(l_lon))``.
    """
    interp = breitkreuz_interpolator(ds, variable=variable, fill_value=fill_value)
    d3, lat3, lon3 = np.meshgrid(np.asarray(l_depth, dtype=float),
                                 np.asarray(l_lat, dtype=float),
                                 np.asarray(l_lon, dtype=float),
                                 indexing='ij')
    points = np.column_stack((d3.ravel(), lat3.ravel(), lon3.ravel()))
    return interp(points).reshape(d3.shape)


def southern_ocean_zonal_mean(field3d, l_lat, lat_max=-40.0):
    """
    Circumpolar (zonal) mean of a ``(depth, lat, lon)`` field over the SO.

    Parameters
    ----------
    field3d : numpy.ndarray
        Array of shape ``(depth, lat, lon)``.
    l_lat : array-like
        Latitude axis matching ``field3d``.
    lat_max : float, optional
        Northern limit of the Southern Ocean (default -40 deg, the definition
        used throughout this project).

    Returns
    -------
    tuple
        ``(zonal_mean, lat_so)`` where ``zonal_mean`` has shape
        ``(depth, n_lat_so)``.
    """
    l_lat = np.asarray(l_lat, dtype=float)
    mask = l_lat <= lat_max
    with warnings.catch_warnings():
        # Latitude bands that are entirely land give an all-NaN slice.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        zonal = np.nanmean(field3d[:, mask, :], axis=2)
    return zonal, l_lat[mask]
