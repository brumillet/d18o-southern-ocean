# -*- coding: utf-8 -*-
"""
Self-checks for :mod:`breitkreuz` against the real PANGAEA NetCDF file.

Run it once after downloading the data, to confirm that the coordinate
normalisation (longitude roll, depth sign flip) and the padding behave as
documented::

    python scripts/test_breitkreuz.py  D:/Data/d18o_so/D18O_Breitkreuz_et_al_2018.nc

The path may also be given through the ``BREITKREUZ_NC`` environment variable.
Exits non-zero if any check fails.
"""

import os
import sys
import warnings

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breitkreuz as bk


_FAILURES = []


def check(name, condition, extra=''):
    """Report a single check and remember failures."""
    print(('PASS  ' if condition else 'FAIL  ') + name + (f'  | {extra}' if extra != '' else ''))
    if not condition:
        _FAILURES.append(name)


def _nanmean_quiet(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def main(nc_path):
    raw = xr.open_dataset(nc_path)
    raw_d18o = raw['D18O_1deg'].values
    raw_lon = raw['lon_1deg_center'].values[0, :]
    raw_annual = _nanmean_quiet(raw_d18o, axis=0)

    ds = bk.load_breitkreuz(nc_path)
    print(ds, '\n')

    # ---- coordinate normalisation ---------------------------------------
    check('dims are (depth, lat, lon)', ds['d18o'].dims == ('depth', 'lat', 'lon'))
    check('depth strictly increasing', np.all(np.diff(ds['depth'].values) > 0))
    check('depth positive downwards', ds['depth'].values[-1] > ds['depth'].values[0] >= 0)
    check('lon strictly increasing', np.all(np.diff(ds['lon'].values) > 0))
    check('lon spans 0-360', ds['lon'].values[0] < 0.5 < 359.5 < ds['lon'].values[-1])
    check('lat unchanged', np.isclose(ds['lat'].values[0], -89.5)
          and np.isclose(ds['lat'].values[-1], 89.5))
    check('padded shape', ds['d18o'].shape == (17, 180, 362), ds['d18o'].shape)

    # ---- padding is a pure copy of the edge cells -----------------------
    v = ds['d18o'].values
    check('top pad copies the first level', np.allclose(v[0], v[1], equal_nan=True))
    check('bottom pad copies the last level', np.allclose(v[-1], v[-2], equal_nan=True))
    check('cyclic pad left == last real column', np.allclose(v[:, :, 0], v[:, :, -2], equal_nan=True))
    check('cyclic pad right == first real column', np.allclose(v[:, :, -1], v[:, :, 1], equal_nan=True))

    # ---- values survive the longitude roll ------------------------------
    for target in (20.5, 219.5):
        raw_target = target if target <= 180 else target - 360
        j_raw = int(np.argmin(np.abs(raw_lon - raw_target)))
        j_new = int(np.argmin(np.abs(ds['lon'].values - target)))
        check(f'values preserved at lon {target}',
              np.allclose(raw_annual[:, :, j_raw], v[1:-1, :, j_new], equal_nan=True))

    check('land stays NaN, not zero',
          np.isnan(v[1:-1, :, 1:-1]).sum() == np.isnan(raw_annual).sum(),
          f'{np.isnan(v[1:-1, :, 1:-1]).sum()} vs {np.isnan(raw_annual).sum()}')

    # ---- month selection -------------------------------------------------
    ds_jan = bk.load_breitkreuz(nc_path, month=1, variables=('D18O',))
    j_raw = int(np.argmin(np.abs(raw_lon - 20.5)))
    j_new = int(np.argmin(np.abs(ds['lon'].values - 20.5)))
    check('month=1 matches raw month index 0',
          np.allclose(raw_d18o[0][:, :, j_raw], ds_jan['d18o'].values[1:-1, :, j_new], equal_nan=True))
    try:
        bk.load_breitkreuz(nc_path, month=13, variables=('D18O',))
        check('month=13 rejected', False)
    except ValueError:
        check('month=13 rejected', True)

    ds_raw_grid = bk.load_breitkreuz(nc_path, pad_vertical=False, pad_cyclic=False,
                                     variables=('D18O',))
    check('unpadded shape', ds_raw_grid['d18o'].shape == (15, 180, 360))

    # ---- interpolator ----------------------------------------------------
    # Pick an abyssal point where the deepest level is genuinely wet, so the
    # bottom padding has something to extend.
    deepest = v[-2]
    so_rows = np.where(ds['lat'].values <= -50)[0]
    wet = [(i, j) for i in so_rows for j in range(1, 361) if np.isfinite(deepest[i, j])]
    i_wet, j_wet = wet[len(wet) // 2]
    lat_wet, lon_wet = ds['lat'].values[i_wet], ds['lon'].values[j_wet]

    interp = bk.breitkreuz_interpolator(ds)
    points = np.array([[2000.0, -55.0, 340.0],      # ordinary deep SO point
                       [10.0, -50.0, 20.0],         # above the top cell centre
                       [5500.0, lat_wet, lon_wet],  # below the bottom cell centre, wet
                       [1000.0, -55.0, 359.9],      # inside the cyclic gap
                       [1000.0, -55.0, 0.1],
                       [5500.0, -55.0, 340.0]])     # below the B18 seafloor
    vals = interp(points)

    check('deep SO value finite and plausible',
          np.isfinite(vals[0]) and -1.0 < vals[0] < 1.0, f'{vals[0]:.3f}')
    check('shallow point recovered by pad_vertical', np.isfinite(vals[1]))
    check('abyssal point over wet seafloor recovered by pad_vertical', np.isfinite(vals[2]))
    check('cyclic gap covered near 360', np.isfinite(vals[3]))
    check('cyclic gap covered near 0', np.isfinite(vals[4]))
    check('point below the B18 seafloor stays NaN', np.isnan(vals[5]))

    unpadded = bk.breitkreuz_interpolator(ds_raw_grid)(points)
    check('without padding the shallow point is NaN', np.isnan(unpadded[1]))
    check('without padding the cyclic gap is NaN', np.isnan(unpadded[3]))
    check('padding leaves interior points untouched', np.isclose(vals[0], unpadded[0]))

    # ---- helpers ---------------------------------------------------------
    tgt_depth, tgt_lat = np.array([500.0, 1500.0, 3000.0]), np.arange(-70.0, -39.0, 2.0)
    tgt_lon = np.arange(0.0, 360.0, 4.0)
    grid = bk.regrid_to(ds, tgt_depth, tgt_lat, tgt_lon)
    check('regrid_to shape', grid.shape == (3, len(tgt_lat), len(tgt_lon)), grid.shape)
    check('regrid_to agrees with the interpolator',
          np.isclose(grid[1, 5, 10],
                     interp(np.array([[1500.0, tgt_lat[5], tgt_lon[10]]]))[0], equal_nan=True))
    check('regrid_to mostly finite over the SO', np.isfinite(grid).mean() > 0.5,
          f'{np.isfinite(grid).mean():.2f} finite')

    zonal, lat_so = bk.southern_ocean_zonal_mean(v[:, :, 1:-1], ds['lat'].values)
    check('zonal mean shape', zonal.shape == (17, int((ds['lat'].values <= -40).sum())))
    check('zonal mean stays south of 40S', lat_so.max() <= -40)
    check('zonal mean values plausible', -0.6 < _nanmean_quiet(zonal, axis=None) < 0.2,
          f'{_nanmean_quiet(zonal, axis=None):.3f}')

    raw.close()

    print()
    if _FAILURES:
        print(f'{len(_FAILURES)} CHECK(S) FAILED: ' + ', '.join(_FAILURES))
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('BREITKREUZ_NC', '')
    if not path:
        sys.exit('Usage: python scripts/test_breitkreuz.py <path to '
                 'D18O_Breitkreuz_et_al_2018.nc>  (or set BREITKREUZ_NC)')
    sys.exit(main(path))
