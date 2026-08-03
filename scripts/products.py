# -*- coding: utf-8 -*-
"""
The gridded delta-18O products, behind one interface.

Each product lives on its own grid, with its own coordinate variable names and
its own vertical axis:

===========  ==================================  ==========================
product      file                                coordinates
===========  ==================================  ==========================
R NEMO       nemo_reconstructed_d18o.nc          lon / lat / depth
R OCIM       ocim_reconstructed_d18o.nc          xt / yt / zt
R TMI        tmi_reconstructed_d18o.nc           xt / yt / zt
LS product   WOCE_climatology_Lg&S_d18o.nc       lon / lat, depth from
                                                 usefull_functions
B18 product  D18O_Breitkreuz_et_al_2018.nc       see :mod:`breitkreuz`
===========  ==================================  ==========================

The notebooks used to spell each of those out by hand, twice over in the case
of ``evaluation_reconstructions``, which is how the second copy ended up
building the NEMO interpolator on the climatology's latitude/longitude vectors
instead of NEMO's own. Loading them through :func:`load_products` keeps the
per-product knowledge in one place and gives every product the same
``sample(depth, lat, lon)`` call.

All products are exposed with the same conventions: values in permil VSMOW on a
``(depth, lat, lon)`` array, depth positive downwards, longitude in 0-360.
"""

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import breitkreuz as bk
import config
import usefull_functions as uf


# Display identity of each product, so that colours and labels stay consistent
# across every figure. Keys are the short names used throughout the notebooks.
PRODUCT_LABELS = {
    'nemo': 'R NEMO',
    'ocim': 'R OCIM',
    'tmi': 'R TMI',
    'ls': 'LS product',
    'b18': bk.BREITKREUZ_LABEL,
}

PRODUCT_COLORS = {
    'nemo': '#1f77b4',
    'ocim': '#ff7f0e',
    'tmi': '#2ca02c',
    'ls': '#d62728',
    'b18': bk.BREITKREUZ_COLOR,
}

# The three water-mass-fraction reconstructions computed in this study, as
# opposed to the two external products.
RECONSTRUCTIONS = ('nemo', 'ocim', 'tmi')
ALL_PRODUCTS = ('nemo', 'ocim', 'tmi', 'ls', 'b18')


@dataclass
class Product:
    """
    One gridded delta-18O product on a regular ``(depth, lat, lon)`` grid.

    Attributes
    ----------
    key : str
        Short name, e.g. ``'ocim'``.
    label : str
        Display name, e.g. ``'R OCIM'``.
    color : str
        Colour used for this product in every figure.
    depth, lat, lon : numpy.ndarray
        1D coordinate axes. Depth is positive downwards, longitude is 0-360.
    d18o : numpy.ndarray
        Values, shape ``(len(depth), len(lat), len(lon))``, permil VSMOW.
    dataset : xarray.Dataset or None
        The dataset the values came from, kept for the extra variables some
        products carry (salinity, neutral density, ...).
    """

    key: str
    label: str
    color: str
    depth: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    d18o: np.ndarray
    dataset: object = None
    _interpolator: object = field(default=None, repr=False)

    @property
    def interpolator(self):
        """Lazily built ``(depth, lat, lon)`` linear interpolator, NaN outside."""
        if self._interpolator is None:
            self._interpolator = RegularGridInterpolator(
                (self.depth, self.lat, self.lon), self.d18o,
                method='linear', bounds_error=False, fill_value=np.nan)
        return self._interpolator

    def sample(self, depth, lat, lon):
        """
        Interpolate the product at arbitrary points.

        Parameters
        ----------
        depth, lat, lon : array-like
            Coordinates of the points. Depth in metres positive downwards,
            longitude in 0-360.

        Returns
        -------
        numpy.ndarray
            Values at those points, NaN where the point falls outside the grid
            or on a land/below-bathymetry cell.
        """
        return self.interpolator((np.asarray(depth, dtype=float),
                                  np.asarray(lat, dtype=float),
                                  np.asarray(lon, dtype=float)))

    def on_grid(self, depth, lat, lon):
        """
        Sample the product onto another regular grid.

        Parameters
        ----------
        depth, lat, lon : array-like
            Target 1D axes.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(depth), len(lat), len(lon))``.
        """
        d3, lat3, lon3 = np.meshgrid(np.asarray(depth, dtype=float),
                                     np.asarray(lat, dtype=float),
                                     np.asarray(lon, dtype=float), indexing='ij')
        return self.sample(d3.ravel(), lat3.ravel(), lon3.ravel()).reshape(d3.shape)


def _from_netcdf(key, path, variable, lon_name, lat_name, depth_name, depth=None):
    """Build a Product from a NetCDF file with named coordinate variables."""
    ds = xr.open_dataset(path)
    values = ds[variable].values.astype(float)
    return Product(
        key=key,
        label=PRODUCT_LABELS[key],
        color=PRODUCT_COLORS[key],
        depth=np.asarray(ds[depth_name].values if depth is None else depth, dtype=float),
        lat=ds[lat_name].values.astype(float),
        lon=ds[lon_name].values.astype(float),
        d18o=values,
        dataset=ds,
    )


def load_product(key, data_path=None, **breitkreuz_kwargs):
    """
    Load a single product by short name.

    Parameters
    ----------
    key : str
        One of ``'nemo'``, ``'ocim'``, ``'tmi'``, ``'ls'``, ``'b18'``.
    data_path : str, optional
        Override the directory; defaults to :data:`config.DATA_PATH`.
    **breitkreuz_kwargs
        Passed to :func:`breitkreuz.load_breitkreuz` when ``key == 'b18'``
        (e.g. ``month=1``).

    Returns
    -------
    Product
    """
    if data_path is not None:
        config.set_data_path(data_path)

    if key == 'nemo':
        return _from_netcdf('nemo', config.data_file('nemo_d18o'), 'd18o_recons',
                            'lon', 'lat', 'depth')
    if key == 'ocim':
        return _from_netcdf('ocim', config.data_file('ocim_d18o'), 'd18o_recons',
                            'xt', 'yt', 'zt')
    if key == 'tmi':
        return _from_netcdf('tmi', config.data_file('tmi_d18o'), 'd18o_recons',
                            'xt', 'yt', 'zt')
    if key == 'ls':
        # The climatology carries no depth variable; its 45 levels are the ones
        # built by usefull_functions.create_l_depth.
        return _from_netcdf('ls', config.data_file('ls_climatology'), 'seawater_d18O',
                            'lon', 'lat', None, depth=uf.create_l_depth())
    if key == 'b18':
        ds = bk.load_breitkreuz(config.data_file('breitkreuz'), **breitkreuz_kwargs)
        return Product(key='b18', label=PRODUCT_LABELS['b18'], color=PRODUCT_COLORS['b18'],
                       depth=ds['depth'].values, lat=ds['lat'].values, lon=ds['lon'].values,
                       d18o=ds['d18o'].values, dataset=ds)

    raise KeyError(f'Unknown product {key!r}; known products: {list(ALL_PRODUCTS)}')


def load_products(keys=ALL_PRODUCTS, data_path=None, **breitkreuz_kwargs):
    """
    Load several products at once.

    Parameters
    ----------
    keys : sequence of str, optional
        Which products to load; defaults to all five.
    data_path : str, optional
        Override the directory; defaults to :data:`config.DATA_PATH`.
    **breitkreuz_kwargs
        Passed through to the B18 loader.

    Returns
    -------
    dict of str to Product
        Keyed by short name, in the order given.

    Examples
    --------
    >>> products = load_products()                       # doctest: +SKIP
    >>> products['ocim'].sample([2000], [-55], [340])    # doctest: +SKIP
    """
    if data_path is not None:
        config.set_data_path(data_path)
    return {key: load_product(key, **(breitkreuz_kwargs if key == 'b18' else {}))
            for key in keys}


def load_climatology(data_path=None):
    """
    Open the WOCE / LeGrande & Schmidt climatology and its interpolators.

    Besides delta-18O this file carries the fields the notebooks use to
    characterise water masses: neutral density, absolute salinity and
    preformed salinity.

    Parameters
    ----------
    data_path : str, optional
        Override the directory; defaults to :data:`config.DATA_PATH`.

    Returns
    -------
    tuple
        ``(dataset, interpolators)`` where ``interpolators`` maps a variable
        name of the dataset to a ``(depth, lat, lon)`` interpolator. Variables
        absent from the file are simply omitted.
    """
    if data_path is not None:
        config.set_data_path(data_path)

    ds = xr.open_dataset(config.data_file('ls_climatology'))
    depth = np.asarray(uf.create_l_depth(), dtype=float)
    lat, lon = ds['lat'].values, ds['lon'].values

    interpolators = {}
    for name in ('seawater_d18O', 'gamma', 'absolute_salinity', 'preformed_salinity'):
        if name in ds:
            interpolators[name] = RegularGridInterpolator(
                (depth, lat, lon), ds[name].values, bounds_error=False)
    return ds, interpolators
