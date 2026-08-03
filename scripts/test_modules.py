# -*- coding: utf-8 -*-
"""
Self-checks for config, products, observations, regions and plotting.

Run from the repository root::

    python scripts/test_modules.py

The checks that need the Breitkreuz NetCDF are skipped unless the data
directory holds it; point at it with the ``D18O_DATA_PATH`` environment
variable, or pass the directory as the first argument::

    python scripts/test_modules.py D:/Data/d18o_so/

Exits non-zero if any check fails.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config, observations as obs, plotting as pl, products as pr, regions as rg  # noqa

ok = True
def check(name, cond, extra=""):
    global ok
    detail = "" if isinstance(extra, str) and extra == "" else f"  | {extra}"
    print(("PASS  " if cond else "FAIL  ") + name + detail)
    if not cond:
        ok = False

# ------------------------------------------------------------------ config
if len(sys.argv) > 1:
    config.set_data_path(sys.argv[1])
check("set_data_path appends a separator",
      config.set_data_path("/tmp/somewhere").endswith("/"))
check("set_data_path keeps an existing separator",
      config.set_data_path("/tmp/somewhere/") == "/tmp/somewhere/")

config.set_data_path(sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH)
HAS_B18 = os.path.exists(config.DATA_PATH + config.FILES["breitkreuz"])
if not HAS_B18:
    print(f"NOTE  {config.FILES['breitkreuz']} not in {config.DATA_PATH}; "
          f"skipping the checks that need it\n")

try:
    config.data_file("a file that is not there")
    check("missing file raises", False)
except KeyError:
    check("unknown key raises KeyError before touching the disk", True)

config.FILES["__probe__"] = "definitely-not-a-real-file.nc"
try:
    config.data_file("__probe__")
    check("missing file raises", False)
except FileNotFoundError as exc:
    check("missing file raises FileNotFoundError naming DATA_PATH",
          "config.DATA_PATH" in str(exc) and "definitely-not-a-real-file.nc" in str(exc))
del config.FILES["__probe__"]
fd = config.figure_dir("evaluation")
check("figure_dir exists and ends with a separator", os.path.isdir(fd) and fd[-1] in "/\\", fd)

# ---------------------------------------------------------------- products
if HAS_B18:
    b18 = pr.load_product("b18")
    check("b18 label/colour", b18.label == "B18 product" and b18.color == "#9467bd")
    check("b18 array shape", b18.d18o.shape == (len(b18.depth), len(b18.lat), len(b18.lon)),
          b18.d18o.shape)
    v = b18.sample([2000.0, 3000.0], [-55.0, -60.0], [340.0, 180.0])
    check("Product.sample returns finite deep SO values", np.all(np.isfinite(v)), v)
    grid = b18.on_grid([1000.0, 2000.0], np.arange(-70.0, -40.0, 5.0), np.arange(0.0, 360.0, 30.0))
    check("Product.on_grid shape", grid.shape == (2, 6, 12), grid.shape)
    check("on_grid agrees with sample",
          np.isclose(grid[0, 2, 3], b18.sample([1000.0], [-60.0], [90.0])[0], equal_nan=True))
    check("interpolator is cached", b18.interpolator is b18.interpolator)
    check("b18 month kwarg reaches the loader",
          pr.load_product("b18", month=3).dataset.attrs["month"] == "month 3")
try:
    pr.load_product("bogus")
    check("unknown product raises", False)
except KeyError:
    check("unknown product raises KeyError", True)
check("PRODUCT_COLORS covers every product",
      set(pr.PRODUCT_COLORS) == set(pr.ALL_PRODUCTS) == set(pr.PRODUCT_LABELS))
check("colours are all distinct", len(set(pr.PRODUCT_COLORS.values())) == 5)

# ------------------------------------------------------------ observations
check("to_0_360 wraps negatives", np.allclose(obs.to_0_360([-140.5, 20.5, 359.5]),
                                              [219.5, 20.5, 359.5]))
#                    deep SO Atl, deep SO Pac, SO Indian, N Pacific, S Indian
df = pd.DataFrame({"Depth": [500.0, 1500.0, 3000.0, 4000.0, 2000.0],
                   "Latitude": [-55.0, -60.0, -45.0, 10.0, -30.0],
                   "Longitude": [340.0, 180.0, 60.0, 200.0, 60.0],
                   "d18O": [-0.1, -0.2, -0.15, 0.1, -0.05]})
if HAS_B18:
    obs.add_product_columns(df, {"b18": b18}, ("Depth", "Latitude", "Longitude"))
    check("add_product_columns default name", "d18O_b18" in df.columns, list(df.columns))
    obs.add_product_columns(df, {"b18": b18}, ("Depth", "Latitude", "Longitude"),
                            columns={"b18": "d18O_B18"})
    check("add_product_columns honours an explicit name", "d18O_B18" in df.columns)
    check("columns agree", np.allclose(df["d18O_b18"], df["d18O_B18"], equal_nan=True))
    check("sampled values are finite in the SO", np.isfinite(df["d18O_B18"][:3]).all(),
          df["d18O_B18"].values)

# ----------------------------------------------------------------- regions
masks = rg.basin_masks(df)
check("basin_masks keys", set(masks) == {"Global", "Southern Ocean", "Atlantic Ocean",
                                         "Indian Ocean", "Pacific Ocean"})
check("SO mask takes everything at or south of 40S",
      list(masks["Southern Ocean"]) == [True, True, True, False, False],
      list(masks["Southern Ocean"]))
check("Global mask takes everything", masks["Global"].all())
check("Indian mask takes 60E at 30S but not at 45S",
      list(masks["Indian Ocean"]) == [False, False, False, False, True],
      list(masks["Indian Ocean"]))
check("Pacific mask takes 200E at 10N",
      list(masks["Pacific Ocean"]) == [False, False, False, True, False],
      list(masks["Pacific Ocean"]))
check("Atlantic mask takes none of these (all are south of 40S or in another basin)",
      not masks["Atlantic Ocean"].any(), list(masks["Atlantic Ocean"]))
# the Atlantic mask must wrap the prime meridian rather than span 30-295
atl = rg.basin_masks(pd.DataFrame({"Latitude": [-20.0, -20.0, -20.0],
                                   "Longitude": [340.0, 10.0, 150.0]}))["Atlantic Ocean"]
check("Atlantic mask wraps the prime meridian", list(atl) == [True, True, False], list(atl))
levels = rg.error_depth_levels([0.0, 10.0, 50.0])
check("error_depth_levels layout", levels[:3] == [0.0, 10.0, 50.0] and levels[3] == 1000
      and levels[-1] == 5900 and len(levels) == 53, (len(levels), levels[-1]))

gdf = pd.DataFrame({"gamma": [27.95, 28.5, 27.95], "absolute_salinity": [34.9, 34.9, 34.1]})
check("na_influenced needs both criteria", list(rg.na_influenced(gdf)) == [True, False, False])

# ---------------------------------------------------------------- plotting
# A synthetic Southern Ocean field, with land holes, so these checks do not
# depend on any data file being present.
so_lat = np.arange(-79.5, -39.0, 1.0)
so_lon = np.arange(0.5, 360.0, 1.0)
rng = np.random.default_rng(0)
field = (-0.1 - 0.002 * np.arange(15)[:, None, None]
         + 0.05 * np.cos(np.deg2rad(so_lat))[None, :, None]
         + 0.01 * rng.standard_normal((15, len(so_lat), len(so_lon))))
field[rng.random(field.shape) < 0.3] = np.nan            # land / below bathymetry

zm = pl.zonal_mean(field)
check("zonal_mean shape", zm.shape == (15, len(so_lat)), zm.shape)
am = pl.area_mean(field, so_lat)
check("area_mean shape", am.shape == (15,), am.shape)

# area weighting must differ from a plain mean, and match a hand computation
plain = np.nanmean(field[8])
check("area_mean differs from the unweighted mean", not np.isclose(am[8], plain),
      f"{am[8]:.5f} vs {plain:.5f}")
layer = field[8]
w = np.broadcast_to(np.cos(np.deg2rad(so_lat))[:, None], layer.shape)
f = np.isfinite(layer)
manual = np.sum(np.where(f, layer, 0) * np.where(f, w, 0)) / np.sum(np.where(f, w, 0))
check("area_mean matches a hand computation", np.isclose(am[8], manual), (am[8], manual))

if HAS_B18:
    b18_so = b18.d18o[:, b18.lat <= -40, :]
    b18_am = pl.area_mean(b18_so, b18.lat[b18.lat <= -40])
    check("area_mean of the real B18 field is plausible",
          -0.5 < np.nanmean(b18_am) < 0.1, np.nanmean(b18_am))

# a level that is entirely NaN must give NaN, not a warning or a zero
empty = np.full((2, 4, 5), np.nan)
with warnings.catch_warnings():
    warnings.simplefilter("error", category=RuntimeWarning)
    check("area_mean handles an all-NaN level", np.all(np.isnan(pl.area_mean(empty, [-70, -60, -50, -45]))))
    check("zonal_mean handles an all-NaN level", np.all(np.isnan(pl.zonal_mean(empty))))

rdf = pd.DataFrame({"Depth": [100.0, 1000.0, 2000.0, 3000.0],
                    "d18O": [0.0, 0.0, 0.0, 0.0],
                    "p1": [0.1, 0.1, 0.1, 0.1],
                    "p2": [0.0, 0.2, 0.2, 0.2]})
rmse, npts = pl.cumulative_rmse(rdf, "d18O", ["p1", "p2"], [0.0, 1000.0, 2500.0])
check("cumulative_rmse shape", rmse.shape == (3, 2), rmse.shape)
check("cumulative_rmse counts samples at or below each level",
      list(npts) == [4, 3, 1], npts)
check("cumulative_rmse constant offset", np.allclose(rmse[:, 0], 0.1), rmse[:, 0])
check("cumulative_rmse below 1000 m", np.isclose(rmse[1, 1], 0.2), rmse[1, 1])
check("cumulative_rmse at the surface mixes both values",
      np.isclose(rmse[0, 1], np.sqrt((0 + 3 * 0.04) / 4)), rmse[0, 1])

fig, ax = plt.subplots()
ax.set_xlim(-3, 7); ax.set_ylim(500, 0)
pl.panel_label(ax, 0, "test", color="red")
pl.panel_label(ax, "(z)")
texts = [t.get_text() for t in ax.texts]
check("panel_label writes letter and title", texts == ["(a)", "test", "(z)"], texts)
check("panel_label uses axes coordinates",
      all(t.get_transform() is ax.transAxes for t in ax.texts))
plt.close(fig)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
