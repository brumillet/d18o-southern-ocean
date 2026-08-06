# -*- coding: utf-8 -*-
"""
Per-longitude North Atlantic (NA) dye maximum, and tracers sampled there.

For each 5-degree longitude bin around the Southern Ocean, this finds the
grid cell where a model's ``DyeNA`` fraction is maximal (within the deep,
sub-polar window used throughout the notebooks: lat -60..-30, a few degrees
either side of the bin's longitude), then reads any tracer defined on that
same grid at exactly that cell.

The d18O and salinity reconstructions this module samples (``d18o_recons``,
``salinity_recons``) are themselves ``sum(fraction_dye * end_member_dye)``,
built in ``optimization_fractions.ipynb``.
"""

from dataclasses import dataclass, field

import numpy as np

LON_BINS = tuple(range(0, 360, 5))
LAT_MIN, LAT_MAX = -60.0, -30.0
LON_HALF_WIDTH = 2.0


@dataclass
class ModelGrid:
    """
    One model's grid, for locating and sampling water-mass extrema on it.

    Attributes
    ----------
    depth, lat, lon : ndarray
        The model's own 1D coordinate axes, depth positive downwards.
    depth_offset : int
        Number of shallow levels excluded from the search (each model was
        trimmed to a different depth range in the original analysis -- NEMO
        at index 46, OCIM at 25, TMI at 18 -- to keep the NA maximum in deep
        water).
    """

    depth: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    depth_offset: int = 0
    _lon3d: np.ndarray = field(default=None, repr=False, compare=False)
    _lat3d: np.ndarray = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.depth = np.asarray(self.depth, dtype=float)
        self.lat = np.asarray(self.lat, dtype=float)
        self.lon = np.asarray(self.lon, dtype=float)
        self._lon3d, self._lat3d = np.meshgrid(self.lon, self.lat)

    @property
    def truncated_depth(self):
        return self.depth[self.depth_offset:]

    def lon_lat_mask(self, lon_center, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_half_width=LON_HALF_WIDTH):
        """2D ``(lat, lon)`` boolean mask for the search window used by :func:`locate_na_max`."""
        return ((self._lon3d >= lon_center - lon_half_width) & (self._lon3d <= lon_center + lon_half_width)
                & (self._lat3d >= lat_min) & (self._lat3d <= lat_max))

    def sample(self, values, flat_index):
        """
        Read a ``(depth, lat, lon)`` array defined on this grid at
        ``flat_index`` (as returned by :func:`locate_na_max`) -- direct
        indexing, no interpolation.
        """
        values = np.asarray(values)
        return values[self.depth_offset:].reshape(-1)[flat_index]


def locate_na_max(grid, dye_na, lon_bins=LON_BINS, lat_min=LAT_MIN, lat_max=LAT_MAX,
                   lon_half_width=LON_HALF_WIDTH):
    """
    For each longitude bin, find the grid cell where ``dye_na`` is maximal
    within ``[lat_min, lat_max]`` and ``[lon_center - w, lon_center + w]``,
    among the depth levels kept by ``grid.depth_offset``.

    Parameters
    ----------
    grid : ModelGrid
    dye_na : ndarray, shape (len(grid.depth), len(grid.lat), len(grid.lon))
        The North Atlantic water-mass fraction (``DyeNA``).

    Returns
    -------
    coords : ndarray, shape (len(lon_bins), 3)
        Physical ``(depth, lat, lon)`` of the maximum, per bin.
    flat_index : ndarray of int, shape (len(lon_bins),)
        Index into the depth-truncated array flattened in C order; reusable
        with :meth:`ModelGrid.sample` to read any other tracer on this grid
        at the same cells.
    """
    values = np.asarray(dye_na)[grid.depth_offset:]
    depth_t = grid.truncated_depth

    coords = np.empty((len(lon_bins), 3))
    flat_index = np.empty(len(lon_bins), dtype=int)
    for i, lon_center in enumerate(lon_bins):
        mask = grid.lon_lat_mask(lon_center, lat_min, lat_max, lon_half_width)
        idx = np.nanargmax(np.where(mask, values, np.nan))
        flat_index[i] = idx
        kz, ilat, jlon = np.unravel_index(idx, values.shape)
        coords[i] = (depth_t[kz], grid.lat[ilat], grid.lon[jlon])
    return coords, flat_index


def reshape_for_so_plot(values, n_shift=60):
    """
    Re-order the last axis (longitude bins, 0..360 step 5 -> 72 bins) so the
    0/360 discontinuity falls away from the Southern Ocean NA signal --
    longitude then runs -60..300, matching every SO figure in this notebook.
    """
    values = np.asarray(values)
    n = values.shape[-1]
    reshaped = values.copy()
    reshaped[..., (n - n_shift):] = values[..., :n_shift]
    reshaped[..., :(n - n_shift)] = values[..., n_shift:]
    return reshaped


def reshaped_lon_axis(lon_bins=LON_BINS, n_shift=60):
    """The longitude axis matching :func:`reshape_for_so_plot`, wrapped to -60..300."""
    lon = reshape_for_so_plot(np.array(lon_bins, dtype=float), n_shift=n_shift)
    lon[lon >= 300] -= 360
    return lon
