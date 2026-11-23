# -*- coding: utf-8 -*-
"""
NADW dilution along the ACC: map, NA-dye sections, and fractions / d18O along the NA core.

Answers the reviewer point that the evolution of NADW "along streamlines" is
not clear. The figure follows the core of North Atlantic (NA) water eastward
around the Southern Ocean:

* (a) south-polar map of the OCIM NA fraction at ~2500 m, with each model's
  NA-maximum path (the locus from :mod:`na_max_reconstruction`, i.e. the
  deep sub-polar core between 60 S and 30 S) and the chosen sections;
* (b) one latitude-depth section of the OCIM NA fraction per longitude, with
  the NA core of each model marked;
* (c) AA / NA / HS fractions at each model's NA core against longitude;
* (d) reconstructed d18O at each model's NA core against longitude, with the
  GISS observations of the NA density / salinity window.

Run from the repository root::

    python scripts/fig_nadw_streamlines.py --data E:/Data/d18o_so/

The observations need pytables (the ``d18o_SO`` conda environment has it).
The figure goes to ``figures/evaluation/NADW_ACC_streamlines.png``.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import na_max_reconstruction as namax  # noqa: E402
import usefull_functions as uf  # noqa: E402
import observations as obs  # noqa: E402
import products as pr  # noqa: E402
import regions as rg  # noqa: E402

# Section longitudes (deg E, 0-360), ordered downstream from the SW Atlantic.
# Must be multiples of 5 to coincide with na_max_reconstruction.LON_BINS.
SECTIONS = (330, 20, 80, 150, 230)
SECTION_TAGS = '12345678'

MAP_DEPTH = 2500.0              # m, depth of the map's NA-fraction slice
SECTION_LAT = (-72, -30)
MAP_SECTION_LAT = (-66, -30)    # drawn part of the section lines on the map

MODELS = ('nemo', 'ocim', 'tmi')
COLORS = {k: pr.PRODUCT_COLORS[k] for k in MODELS}
LABELS = {k: pr.PRODUCT_LABELS[k] for k in MODELS}                  # 'R NEMO', ...
DEPTH_OFFSET = {'nemo': 46, 'ocim': 25, 'tmi': 18}                   # as in evaluation notebook

D18O_LABEL = r'$\delta^{18}O_{sw}$ (‰ VSMOW)'
DYE_LABEL = 'NA water-mass fraction (–)'

# Neutral palette for the NA fraction, so the coloured model cores stay legible on top.
DYE_CMAP = ListedColormap(plt.get_cmap('Greys')(np.linspace(0.04, 0.72, 256)))
LAND_COLOR = '#c8b591'       # continents
SHALLOW_COLOR = '#ece3d0'    # ocean, but seafloor shallower than the map depth
DYE_RANGE = (0.3, 0.7)       # colour range of the NA fraction; values outside saturate
HALO = [pe.Stroke(linewidth=3.2, foreground='w'), pe.Normal()]

# Deep water masses at the NA core, panel (c).
DEEP_DYES = ('AA', 'NA', 'HS')
DYE_VARS = {'nemo': 'Dye{}', 'ocim': 'dye{}_steady', 'tmi': 'dye{}'}
DYE_LS = {'AA': '--', 'NA': '-', 'HS': '-.'}
DYE_LW = {'AA': 1.2, 'NA': 1.8, 'HS': 1.2}
OBS_COLOR = '0.25'

TEXT_SCALE = 1.2             # all text but the section headers
SECTION_TITLE_SIZE = 12


def fs(size):
    return size * TEXT_SCALE


# ----------------------------------------------------------------------------- data
def _axes(ds):
    """(depth, lat, lon) 1D axes, whatever the file calls them."""
    if 'zt' in ds:
        return ds['zt'].values, ds['yt'].values, ds['xt'].values
    return ds['depth'].values.astype(float), ds['lat'].values, ds['lon'].values


def load_model(key):
    """NA dye and reconstructed d18O of one model on its own grid, plus the values along its NA core."""
    fkey, dye_var = {'nemo': ('nemo_fractions', 'DyeNA'),
                     'ocim': ('ocim_fractions', 'dyeNA_steady'),
                     'tmi': ('tmi_fractions', 'dyeNA')}[key]
    dye_ds = xr.open_dataset(config.data_file(fkey))
    depth, lat, lon = _axes(dye_ds)
    out = dict(depth=depth, lat=lat, lon=lon, dye=dye_ds[dye_var].values,
               d18o=xr.open_dataset(config.data_file(key + '_d18o'))['d18o_recons'].values)
    g = namax.ModelGrid(depth, lat, lon, depth_offset=DEPTH_OFFSET[key])
    out['core'], idx = namax.locate_na_max(g, out['dye'])   # (72, [depth, lat, lon])

    # every tracer read at the NA-core cell of each 5-degree bin
    out['along'] = {'d18o': g.sample(out['d18o'], idx), 'NA': g.sample(out['dye'], idx)}
    for dye in DEEP_DYES:
        if dye != 'NA':
            out['along'][dye] = g.sample(dye_ds[DYE_VARS[key].format(dye)].values, idx)
    return out


def load_observations():
    """
    GISS d18O observations in the NA density and salinity window (as in the
    evaluation notebook), with a ``Longitude2`` column wrapped to -67..293
    like the along-ACC axis.
    """
    d = obs.load_giss('modified')
    so = d[rg.southern_ocean(d)].copy()
    aux = so.where((so['gamma_n'] <= rg.SO_GAMMA_MAX) & (so['gamma_n'] >= rg.SO_GAMMA_MIN)).dropna()
    window = rg.na_influenced(so, gamma_col='gamma_n')
    # the notebook's 12 datasets, so the selection is the same as in its Figure 8
    out = pd.concat([df.where(window).dropna() for df in uf.create_dfs(aux)], ignore_index=True)
    out['Longitude2'] = np.where(out['Longitude'] <= 293, out['Longitude'], out['Longitude'] - 360)
    return out


def core_value(m, ibin, var):
    """`var` at the NA-core cell of longitude bin `ibin` (depth, value)."""
    cdep, clat, clon = m['core'][ibin]
    k = int(np.argmin(np.abs(m['depth'] - cdep)))
    ilat = int(np.argmin(np.abs(m['lat'] - clat)))
    ilon = int(np.argmin(np.abs(m['lon'] - clon)))
    return cdep, m[var][k, ilat, ilon]


def section(m, lon_center, var='dye'):
    """Latitude-depth section of `var`, averaged over the +-2 deg window used to find the core."""
    sel = np.abs(((m['lon'] - lon_center + 180) % 360) - 180) <= namax.LON_HALF_WIDTH
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(m[var][:, :, sel], axis=2)


# ----------------------------------------------------------------------------- panels
GRID_COLOR = '0.25'
GRID_LATS = (-75, -60, -45, -30)
GRID_LONS = (0, 60, 120, 180, 240, 300)
LAT_LABEL_LON = 195.0           # meridian along which the latitude circles are labelled


def _circle_boundary(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)


def _lon_text(lon):
    lon = lon % 360
    if lon in (0, 180):
        return f'{lon:.0f}°'
    return f'{lon:.0f}°E' if lon < 180 else f'{360 - lon:.0f}°W'


def _map_grid(ax):
    """Latitude circles and meridians, drawn and labelled by hand so they stay legible on the shading."""
    pc = ccrs.PlateCarree()
    lon = np.linspace(0, 360, 361)
    for la in GRID_LATS:
        ax.plot(lon, np.full_like(lon, la), color=GRID_COLOR, lw=0.7, ls='--', transform=pc, zorder=3)
        ax.text(LAT_LABEL_LON, la + 1.2, f'{-la}°S', transform=pc, fontsize=fs(8), color=GRID_COLOR,
                ha='center', va='bottom', zorder=8,
                bbox=dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.75))
    for lo in GRID_LONS:
        ax.plot([lo, lo], [-80, -25], color=GRID_COLOR, lw=0.7, ls='--', transform=pc, zorder=3)
    # longitude labels just outside the disk
    # (SouthPolarStereo(central_longitude=0): 0 deg at the top, east clockwise)
    for lo in GRID_LONS:
        ang = np.deg2rad(lo)
        x, y = 0.5 + 0.55 * np.sin(ang), 0.5 + 0.55 * np.cos(ang)
        ax.text(x, y, _lon_text(lo), transform=ax.transAxes, fontsize=fs(8), ha='center', va='center',
                clip_on=False)


def plot_map(ax, models, sections):
    ocim = models['ocim']
    k = int(np.argmin(np.abs(ocim['depth'] - MAP_DEPTH)))
    field = ocim['dye'][k]
    lon_c = np.append(ocim['lon'], ocim['lon'][0] + 360)
    field_c = np.concatenate([field, field[:, :1]], axis=1)
    # pale sand everywhere first: shows where the seafloor is shallower than MAP_DEPTH
    ax.pcolormesh(lon_c, ocim['lat'], np.zeros_like(field_c), cmap=ListedColormap([SHALLOW_COLOR]),
                  transform=ccrs.PlateCarree(), shading='auto', rasterized=True)
    pc = ax.pcolormesh(lon_c, ocim['lat'], field_c, cmap=DYE_CMAP, vmin=DYE_RANGE[0], vmax=DYE_RANGE[1],
                       transform=ccrs.PlateCarree(), shading='auto', rasterized=True)
    # land / no-data from the OCIM surface mask (no shapefile download needed)
    land = np.where(np.isnan(np.concatenate([ocim['dye'][0], ocim['dye'][0][:, :1]], axis=1)), 1.0, np.nan)
    ax.pcolormesh(lon_c, ocim['lat'], land, cmap=ListedColormap([LAND_COLOR]),
                  transform=ccrs.PlateCarree(), shading='auto', rasterized=True)
    ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
    _circle_boundary(ax)
    _map_grid(ax)

    # NA-core path of each model
    for key in MODELS:
        c = models[key]['core']
        lon = np.append(c[:, 2], c[0, 2] + 360)
        lat = np.append(c[:, 1], c[0, 1])
        ax.plot(lon, lat, color=COLORS[key], lw=1.8, path_effects=HALO,
                transform=ccrs.PlateCarree(), zorder=4 if key == 'ocim' else 4.5)

    # section lines and tags
    for tag, lon0 in zip(SECTION_TAGS, sections):
        ax.plot([lon0, lon0], MAP_SECTION_LAT, color='w', lw=2.4, transform=ccrs.PlateCarree(), zorder=5)
        ax.plot([lon0, lon0], MAP_SECTION_LAT, color='k', lw=1.2, transform=ccrs.PlateCarree(), zorder=6)
        ax.text(lon0, SECTION_LAT[1] + 4, tag, transform=ccrs.PlateCarree(), ha='center', va='center',
                fontsize=fs(12), fontweight='bold', zorder=7,
                bbox=dict(boxstyle='circle,pad=0.2', fc='w', ec='k', lw=0.8))
    return pc


def plot_section(ax, models, lon0, ibin, show_ylabel):
    ocim = models['ocim']
    sec = section(ocim, lon0)
    lev = np.linspace(DYE_RANGE[0], DYE_RANGE[1], 9)
    cf = ax.contourf(ocim['lat'], ocim['depth'], sec, levels=lev, cmap=DYE_CMAP, extend='both')
    cs = ax.contour(ocim['lat'], ocim['depth'], sec, levels=[0.3, 0.5, 0.7], colors='0.15', linewidths=0.6)
    ax.clabel(cs, fmt='%.1f', fontsize=fs(7))
    ax.set_facecolor(LAND_COLOR)
    for key in MODELS:
        d, la, _ = models[key]['core'][ibin]
        ax.plot(la, d, marker='*' if key == 'ocim' else 'o', ms=11 if key == 'ocim' else 6,
                mfc=COLORS[key], mec='k', mew=0.7, ls='none', zorder=5)
    ax.set_xlim(*SECTION_LAT)
    ax.set_ylim(5500, 0)
    ax.set_xticks([-70, -60, -50, -40, -30])
    ax.set_xticklabels(['70°S', '60°S', '50°S', '40°S', '30°S'], fontsize=fs(8))
    ax.tick_params(axis='y', labelsize=fs(8))
    if show_ylabel:
        ax.set_ylabel('Depth (m)', fontsize=fs(9))
    else:
        ax.set_yticklabels([])
    return cf


def _along_lon(lon0):
    """Section longitude (0-360) on the -60..300 along-ACC axis."""
    return lon0 - 360 if lon0 >= 300 else lon0


def _along_axis(ax, sections):
    for tag, lon0 in zip(SECTION_TAGS, sections):
        x = _along_lon(lon0)
        ax.axvline(x, color='0.35', lw=0.8, ls=':', zorder=0)
        ax.text(x, 1.0, tag, transform=ax.get_xaxis_transform(), ha='center', va='center',
                fontsize=fs(9), fontweight='bold', clip_on=False,
                bbox=dict(boxstyle='circle,pad=0.15', fc='w', ec='k', lw=0.7))
    ax.set_xlim(-65, 300)
    ax.set_xticks(range(-60, 301, 60))
    ax.set_xlabel('Longitude (°E)', fontsize=fs(9))
    ax.tick_params(labelsize=fs(8))


def plot_along_dyes(ax, models, sections):
    """AA / NA / HS fractions at each model's NA core, against longitude."""
    lon = namax.reshaped_lon_axis()
    for key in MODELS:
        for dye in DEEP_DYES:
            ax.plot(lon, namax.reshape_for_so_plot(models[key]['along'][dye]), color=COLORS[key],
                    ls=DYE_LS[dye], lw=DYE_LW[dye])
    _along_axis(ax, sections)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Water-mass fraction (–)', fontsize=fs(9))
    leg = ax.legend([Line2D([0], [0], color='0.4', lw=DYE_LW[d], ls=DYE_LS[d]) for d in DEEP_DYES],
                    DEEP_DYES, fontsize=fs(10), loc='upper right', frameon=False, ncol=3)
    ax.add_artist(leg)
    # the models' own fractions here, not reconstructions: no "R" prefix
    ax.legend([Line2D([0], [0], color=COLORS[k], lw=1.8) for k in MODELS],
              [LABELS[k].removeprefix('R ') for k in MODELS],
              fontsize=fs(10), loc='upper right', bbox_to_anchor=(1.0, 0.88), frameon=False, ncol=3)


def plot_along_d18o(ax, models, observations, sections):
    """Reconstructed d18O at each model's NA core, with the observations of the NA window."""
    ax.scatter(observations['Longitude2'], observations['d18O'], marker='o', s=14, color=OBS_COLOR,
               lw=0, zorder=2)
    lon = namax.reshaped_lon_axis()
    for key in MODELS:
        ax.plot(lon, namax.reshape_for_so_plot(models[key]['along']['d18o']), color=COLORS[key], lw=1.8,
                zorder=3)
    _along_axis(ax, sections)
    ax.set_ylim(-0.3, 0.3)
    ax.set_ylabel(D18O_LABEL, fontsize=fs(9))
    ax.legend([Line2D([0], [0], color=OBS_COLOR, marker='o', ls='none', ms=4.5, mew=0)],
              ['GISS obs., NA window'], fontsize=fs(10), loc='upper right', frameon=False, handlelength=0.8)


# ----------------------------------------------------------------------------- figure
def _row_letter(fig, ax, letter):
    fig.text(0.008, ax.get_position().y1, f'({letter})', fontsize=fs(12), fontweight='bold', va='top')


def make_figure(models, observations, sections=SECTIONS):
    n = len(sections)
    ibins = [namax.LON_BINS.index(s) for s in sections]
    ratios = [4.6, 2.5, 3.6]
    fig = plt.figure(figsize=(2.6 * n + 0.8, sum(ratios)))
    gs = fig.add_gridspec(len(ratios), n, height_ratios=ratios, hspace=0.5, wspace=0.2,
                          left=0.07, right=0.9, top=0.97, bottom=0.06)

    # (a) map
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.SouthPolarStereo(central_longitude=0))
    pc = plot_map(ax_map, models, sections)
    mpos = ax_map.get_position()
    half_w = 0.5 * mpos.height * fig.get_figheight() / fig.get_figwidth()   # disk radius, figure-x units
    cx = mpos.x0 + 0.5 * mpos.width
    cax = fig.add_axes([cx + half_w + 0.06, mpos.y0 + 0.3 * mpos.height, 0.012, 0.5 * mpos.height])
    cb = fig.colorbar(pc, cax=cax, extend='both')
    cb.set_label(f'OCIM {DYE_LABEL}\nat {MAP_DEPTH:.0f} m', fontsize=fs(8))
    cb.ax.tick_params(labelsize=fs(8))
    handles = [Line2D([0], [0], color=COLORS[k], lw=1.6, marker='*' if k == 'ocim' else 'o',
                      mfc=COLORS[k], mec='k', ms=8 if k == 'ocim' else 5) for k in MODELS]
    fig.legend(handles, [f'{LABELS[k]} NA core' for k in MODELS], loc='center right',
               bbox_to_anchor=(cx - half_w - 0.05, mpos.y0 + 0.5 * mpos.height), fontsize=fs(9), frameon=False)
    fig.text(0.008, mpos.y1, '(a)', fontsize=fs(12), fontweight='bold', va='top')

    # (b) sections
    for j, (lon0, ib) in enumerate(zip(sections, ibins)):
        ax = fig.add_subplot(gs[1, j])
        cf = plot_section(ax, models, lon0, ib, show_ylabel=(j == 0))
        ax.set_title(f'Section {SECTION_TAGS[j]} ({_lon_text(lon0)})', fontsize=SECTION_TITLE_SIZE,
                     fontweight='bold', pad=6)
        if j == 0:
            _row_letter(fig, ax, 'b')
    pos = ax.get_position()
    cax = fig.add_axes([0.915, pos.y0, 0.010, pos.height])
    cb = fig.colorbar(cf, cax=cax, ticks=np.linspace(DYE_RANGE[0], DYE_RANGE[1], 5))
    cb.set_label('OCIM NA\nfraction (–)', fontsize=fs(8))
    cb.ax.tick_params(labelsize=fs(8))

    # (c) fractions and (d) d18O along the NA core
    sub = gs[2, :].subgridspec(1, 2, wspace=0.18)
    ax_c, ax_d = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
    plot_along_dyes(ax_c, models, sections)
    plot_along_d18o(ax_d, models, observations, sections)
    _row_letter(fig, ax_c, 'c')
    ax_d.text(-0.13, 1.0, '(d)', transform=ax_d.transAxes, fontsize=fs(12), fontweight='bold',
              ha='left', va='bottom')
    return fig


def print_core_table(models, sections):
    """NA-core values per section, to quote in the text / caption."""
    print(f'{"sec":>4}{"lon":>6} | ' + ' | '.join(f'{k:^23}' for k in MODELS))
    print(f'{"":>10} | ' + ' | '.join(f'{"z(m)  lat  NA   d18O":^23}' for _ in MODELS))
    for tag, lon0 in zip(SECTION_TAGS, sections):
        ib = namax.LON_BINS.index(lon0)
        cells = []
        for k in MODELS:
            m = models[k]
            d, la, _ = m['core'][ib]
            _, dye = core_value(m, ib, 'dye')
            _, d18 = core_value(m, ib, 'd18o')
            cells.append(f'{d:5.0f} {la:5.1f} {dye:4.2f} {d18:6.3f}')
        print(f'{tag:>4}{lon0:>6} | ' + ' | '.join(cells))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--data', default=None, help='data directory (else D18O_DATA_PATH / config default)')
    p.add_argument('--sections', type=int, nargs='+', default=list(SECTIONS),
                   help='section longitudes, deg E 0-360, multiples of 5, in downstream order')
    p.add_argument('--out', default=None, help='output directory (default figures/evaluation)')
    a = p.parse_args()
    if a.data:
        config.set_data_path(a.data)
    out = a.out or config.figure_dir('evaluation')

    models = {k: load_model(k) for k in MODELS}
    print_core_table(models, a.sections)
    fig = make_figure(models, load_observations(), a.sections)
    name = os.path.join(out, 'NADW_ACC_streamlines.png')
    fig.savefig(name, dpi=250)
    print('saved', name)


if __name__ == '__main__':
    main()
