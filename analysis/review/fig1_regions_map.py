# -*- coding: utf-8 -*-
"""
Figure 1 with the surface source regions (reviewer 1, comment on region bounds).

(a) The six surface regions, taken as the dye with the largest fraction in the
    top layer of OCIM (each dye is 1 in its own surface region).
(b) The 12 Southern Ocean datasets, as in the submitted Figure 1, with the
    boundaries between the regions present south of 20 S, labelled with their
    full names (low latitudes, mid- and high-latitude Southern Ocean, Antarctic margin).

Run from the repository root:
    python analysis/review/fig1_regions_map.py --data E:/Data/d18o_so/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import usefull_functions as uf  # noqa: E402

# Ordered from south to north so that contours between consecutive indices
# are the boundaries that matter in the Southern Ocean.
REGIONS = ['AA', 'HS', 'MS', 'LL', 'NP', 'NA']
NAMES = {'AA': 'Antarctic margin (AA)', 'HS': 'High-latitude Southern Ocean (HS)',
         'MS': 'Mid-latitude Southern Ocean (MS)', 'LL': 'Low latitudes (LL)',
         'NP': 'North Pacific (NP)', 'NA': 'North Atlantic (NA)'}
PALE = ['#c6dbef', '#c7e9c0', '#fdd0a2', '#f0f0f0', '#dadaeb', '#fcbba1']
FONT = 13


def surface_regions(path):
    ds = xr.open_dataset(path + 'ocim_steady_dyes.nc')
    lon, lat = ds['xt'].values, ds['yt'].values
    stack = np.stack([ds[f'dye{r}_steady'].isel(Depth=0).values for r in REGIONS])
    ocean = np.isfinite(stack).all(axis=0) & (np.nansum(stack, axis=0) > 0)
    idx = np.where(ocean, np.nanargmax(np.where(np.isfinite(stack), stack, -1), axis=0), np.nan)
    # close the seam at the dateline for plotting
    lon = np.append(lon, lon[0] + 360)
    idx = np.concatenate([idx, idx[:, :1]], axis=1)
    return lon, lat, idx


def load_datasets(path):
    giss = pd.read_table(path + 'giss_d18o.txt')
    cas = pd.read_table(path + 'casimir_neutral_density.txt', header=None,
                        names=['X', 'Y', 'Z', 'd18O', 'CT', 'SA', 'G', 'SP', 'TI', 'P'],
                        index_col=False)
    giss['gamma_n'] = cas['G'].astype(float)
    giss['absolute_salinity'] = cas['SA'].astype(float)
    so = giss[giss['Latitude'] <= -40.0].copy()
    so['Longitude'] = [lon if lon >= 0 else 360 + lon for lon in so['Longitude']]
    so = so.where((so['Depth'] != -999.0) & (so['d18O'] != '**') & (so['Salinity'] != '**'))
    so['d18O'] = so['d18O'].astype(float)
    so['Salinity'] = so['Salinity'].astype(float)
    aux = so.where((so['gamma_n'] <= 28.6) & (so['gamma_n'] >= 27)).dropna()
    return uf.create_dfs(aux)


def main(path, out):
    lon, lat, idx = surface_regions(path)
    dfs = load_datasets(path)

    fig = plt.figure(figsize=(16, 3.4))

    # (b) Southern Ocean datasets with the region boundaries
    ax = fig.add_axes([0.05, 0.05, 0.92, 0.9], projection=ccrs.PlateCarree())
    ax.contour(lon, lat, idx, levels=[0.5, 1.5, 2.5], colors='k', linewidths=1.1, linestyles='--',
               transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=2)
    ax.coastlines(resolution='110m', zorder=3)
    gl = ax.gridlines(draw_labels={'bottom': 'x', 'left': 'y'}, ylocs=[-90 + 15 * i for i in range(5)],
                      linewidth=0.3, color='0.6')
    gl.xlabel_style = {'size': FONT - 1}
    gl.ylabel_style = {'size': FONT - 1}
    ax.set_extent([-180, 180, -80, -20], crs=ccrs.PlateCarree())
    colours = plt.cm.tab20(range(12))
    for i in range(12):
        ax.scatter(dfs[i]['Longitude'], dfs[i]['Latitude'], color=colours[i], marker='x', zorder=4,
                   transform=ccrs.PlateCarree())
        ax.text(uf.coords_dataset[i][0], uf.coords_dataset[i][1], f'({i + 1})', fontsize=FONT,
                color=colours[i], zorder=5, transform=ccrs.PlateCarree())
    # region labels, placed in the Pacific sector where the bands are clear of data
    for name, y in [('LL', -25), ('MS', -37.5), ('HS', -57), ('AA', -70)]:
        ax.text(-118, y, NAMES[name], fontsize=FONT - 1, fontweight='bold', ha='center', va='center', zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5),
                transform=ccrs.PlateCarree())
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('written', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='E:/Data/d18o_so/')
    p.add_argument('--out', default=os.path.join(ROOT, 'figures', 'review', 'fig1_regions_map.png'))
    a = p.parse_args()
    main(a.data, a.out)
