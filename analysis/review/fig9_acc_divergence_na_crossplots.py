# -*- coding: utf-8 -*-
"""
Figure 9: reconstructions at the North-Atlantic-like observations, and the
disagreement between delta-18O products along the ACC.

Top row (a-c): observed against reconstructed delta-18O at the NA-like
observations, one panel per reconstruction (R NEMO, R OCIM, R TMI). The
observations are selected as in notebooks/evaluation_reconstructions.ipynb
(cells 90-92): GISS Southern Ocean compilation (south of 40 S), split into the
12 datasets, kept in the regions.na_influenced window (27.9 <= gamma_n <=
28.05 kg m-3, absolute salinity >= 34.8 g/kg), with gamma_n and absolute
salinity from the climatology.

Bottom (d-e): range (max - min) between products along the ACC (45-65 S,
200-5000 m) as a function of longitude, eastward from Drake Passage, and
neutral density; (d) R NEMO, R OCIM, R TMI; (e) the same plus LS and B18.

Run from the repository root:
    python analysis/review/fig9_acc_divergence_na_crossplots.py --data E:/Data/d18o_so/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config  # noqa: E402
import products as pr  # noqa: E402
import observations as obs  # noqa: E402
import regions as rg  # noqa: E402
import usefull_functions as uf  # noqa: E402
import plotting as pl  # noqa: E402

FONT = 16
RECONS = ('nemo', 'ocim', 'tmi')
ALL = RECONS + ('ls', 'b18')
PRECISION = 0.05          # permil, analytical precision

# ACC section
LAT_S, LAT_N = -65.0, -45.0
DEPTHS = np.arange(200.0, 5001.0, 100.0)
LATS = np.arange(LAT_S + 0.5, LAT_N, 1.0)
LONS = np.arange(0.5, 360.0, 1.0)
LON0 = 290.0              # path starts at Drake Passage, 70 W
L_EDGES = np.arange(0, 361, 10.0)
G_EDGES = np.round(np.arange(26.8, 28.4501, 0.05), 3)
LEVELS = np.round(np.arange(0, 0.2 + 1e-9, 0.025), 3)   # 8 colour classes
NA_COLOR = '#00c2e0'   # cyan: NA core lines, label and the NA-like observations in (d)
PREC_COLOR = '#f08080'    # light red: analytical precision, band in (a-c) and contour in (d-e)

# Sectors along the path coordinate (degrees east of 70 W): limits at 20 E and 150 E
SECTORS = [('Atlantic', 0, 90, '#4c72b0'), ('Indian', 90, 220, '#55a868'), ('Pacific', 220, 360, '#c44e52')]

# neutral-density axis: same stretch as scripts/custom_density_scale.py, correctly inverted
G_TOP = 28.6
fwd = lambda g: -np.log1p(np.clip(G_TOP - np.asarray(g, float), 1e-6, None))   # noqa: E731
inv = lambda y: G_TOP - np.expm1(-np.asarray(y, float))                        # noqa: E731


def sample(p, depth, lat, lon):
    """Sample a product; longitude clamped into its grid (OCIM/TMI stop at 1 and 359 E)."""
    return p.sample(np.asarray(depth, float), np.asarray(lat, float),
                    np.clip(np.asarray(lon, float), p.lon.min(), p.lon.max()))


def sector_of(path):
    out = np.empty(len(path), dtype=object)
    for name, a, b, _ in SECTORS:
        out[(path >= a) & (path < b)] = name
    return out


# --------------------------------------------------------------------------
# NA-like observations (as in evaluation_reconstructions.ipynb)
# --------------------------------------------------------------------------
def na_observations(prods, clim):
    # GISS, cells 39 and 90-92
    d18o = obs.load_giss('modified')
    d18o_SO = d18o[rg.southern_ocean(d18o)].copy()
    obs.add_climatology_columns(d18o_SO, {'gamma': clim['gamma'], 'absolute_salinity': clim['absolute_salinity']},
                                ('Depth', 'Latitude', 'Longitude'), columns={'gamma': 'Gamma'})
    aux_SO = d18o_SO.where((d18o_SO['Gamma'] <= rg.SO_GAMMA_MAX)
                           & (d18o_SO['Gamma'] >= rg.SO_GAMMA_MIN)).dropna()
    na_window = rg.na_influenced(d18o_SO, gamma_col='Gamma')
    giss = pd.concat([df.where(na_window).dropna() for df in uf.create_dfs(aux_SO)])
    na = pd.DataFrame({'d18O': giss['d18O'], 'depth': giss['Depth'], 'lat': giss['Latitude'],
                       'lon': giss['Longitude'] % 360, 'gamma': giss['Gamma']}).reset_index(drop=True)
    for k in RECONS:
        na[k] = sample(prods[k], na.depth, na.lat, na.lon)
    na = na.dropna(subset=['d18O'] + list(RECONS)).reset_index(drop=True)
    na['path'] = (na.lon - LON0) % 360
    na['sector'] = sector_of(na.path.values)
    return na


# --------------------------------------------------------------------------
# ACC section (as in figures/review/acc_density_section.png)
# --------------------------------------------------------------------------
def acc_section(prods, clim):
    D3, LAT3, LON3 = np.meshgrid(DEPTHS, LATS, LONS, indexing='ij')
    pts = (D3.ravel(), LAT3.ravel(), LON3.ravel())
    grid = {k: sample(p, *pts).reshape(D3.shape) for k, p in prods.items()}
    gamma = clim['gamma'](np.column_stack(pts)).reshape(D3.shape)
    ok = np.isfinite(gamma) & np.all([np.isfinite(grid[k]) for k in ALL], axis=0)
    sR = np.stack([grid[k] for k in RECONS])
    sA = np.stack([grid[k] for k in ALL])
    fields = {'R': sR.max(0) - sR.min(0), 'all': sA.max(0) - sA.min(0), 'depth': D3}

    path = (LON3 - LON0) % 360
    W = np.cos(np.deg2rad(LAT3))
    il, ig = np.digitize(path, L_EDGES) - 1, np.digitize(gamma, G_EDGES) - 1
    nl, ng = len(L_EDGES) - 1, len(G_EDGES) - 1
    m = ok & (ig >= 0) & (ig < ng)
    flat = il[m] * ng + ig[m]
    ws = np.bincount(flat, weights=W[m], minlength=nl * ng)
    n = np.bincount(flat, minlength=nl * ng)
    sec = {}
    for key, f in fields.items():
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.bincount(flat, weights=(f * W)[m], minlength=nl * ng) / ws
        out[n < 3] = np.nan
        sec[key] = out.reshape(nl, ng).T          # (gamma, longitude)
    return sec


def giss_in_band(clim):
    """All GISS observations in the ACC band below 200 m, placed at their neutral density."""
    giss = obs.load_giss('modified').dropna(subset=['d18O', 'Depth', 'Latitude', 'Longitude'])
    giss = giss[(giss.Latitude >= LAT_S) & (giss.Latitude <= LAT_N) & (giss.Depth >= DEPTHS[0])].copy()
    xyz = np.column_stack([giss.Depth, giss.Latitude, giss.Longitude])
    giss['gamma_use'] = giss['gamma_n'].fillna(pd.Series(clim['gamma'](xyz), index=giss.index))
    giss = giss.dropna(subset=['gamma_use'])
    giss['path'] = (giss.Longitude - LON0) % 360
    return giss


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def draw_section(ax, field, sec, cmap, norm, label, dots=None, na=None):
    lc = 0.5 * (L_EDGES[1:] + L_EDGES[:-1])
    gc = 0.5 * (G_EDGES[1:] + G_EDGES[:-1])
    im = ax.pcolormesh(L_EDGES, G_EDGES, field, cmap=cmap, norm=norm, shading='flat')
    ax.contour(lc, gc, field, levels=[PRECISION], colors=PREC_COLOR, linewidths=2.2)
    if dots is not None:
        ax.scatter(dots.path, dots.gamma_use, s=7, c='0.9', alpha=0.45, linewidths=0, zorder=5)
    if na is not None:
        # the NA-like observations of (a-c), highlighted in the colour of the NA core lines
        ax.scatter(na.path, na.gamma, s=40, c=NA_COLOR, edgecolors='k', linewidths=0.6, zorder=6)
    ax.set_yscale('function', functions=(fwd, inv))
    ax.set_ylim(G_EDGES[-1], G_EDGES[0])
    yt = np.round(np.arange(26.8, 28.4001, 0.1), 2)
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{g:.1f}' if i % 2 == 0 else '' for i, g in enumerate(yt)])
    ax.set_ylabel(pl.GAMMA_LABEL, fontsize=FONT)
    ax.tick_params(labelsize=FONT - 2)
    for g in (rg.NA_GAMMA_MIN, rg.NA_GAMMA_MAX):
        ax.axhline(g, color=NA_COLOR, lw=2.2, ls='--')
    for x in (90, 220):
        ax.axvline(x, color='k', lw=0.8, ls=':')
    cs = ax.contour(lc, gc, sec['depth'], levels=[1000, 2000], colors='k', linewidths=1.4)
    cs.set(path_effects=[pe.Stroke(linewidth=2.2, foreground='w'), pe.Normal()])
    for t in ax.clabel(cs, fmt=lambda v: f'{v:.0f} m', fontsize=FONT - 3):
        t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='w')])
    for name, a, b, _ in SECTORS:
        ax.text(0.5 * (a + b), 1.01, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=FONT)
    ax.text(1.005, (fwd(np.mean([rg.NA_GAMMA_MIN, rg.NA_GAMMA_MAX])) - fwd(G_EDGES[-1]))
            / (fwd(G_EDGES[0]) - fwd(G_EDGES[-1])), 'NA core', transform=ax.transAxes, va='center',
            rotation=90, fontsize=FONT, color=NA_COLOR, fontweight='bold')
    # panel letter above the top-left corner, level with the sector names
    ax.text(0.0, 1.01, label, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT,
            fontweight='bold')
    ax.set_xlim(0, 360)
    return im


def draw_crossplot(ax, na, key, label, lims, ticks, ylabels):
    ax.fill_between(lims, [lims[0] - PRECISION, lims[1] - PRECISION], [lims[0] + PRECISION, lims[1] + PRECISION],
                    color=PREC_COLOR, alpha=0.35, lw=0, zorder=0)
    ax.plot(lims, lims, color='0.4', ls='--', lw=1, zorder=1)
    for name, _, _, col in SECTORS:
        s = na[na.sector == name]
        if len(s):
            ax.scatter(s.d18O, s[key], s=45, marker='o', c=col, alpha=0.75, linewidths=0, zorder=3)
    err = na[key] - na.d18O
    box = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1)
    ax.text(0.04, 0.84, f'RMSE = {np.sqrt(np.mean(err ** 2)):.3f} ‰',
            transform=ax.transAxes, ha='left', va='top', fontsize=FONT - 2, bbox=box, zorder=5)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=FONT - 2)
    ax.set_yticklabels([f'{t:.1f}' for t in ticks] if ylabels else [], fontsize=FONT - 2)
    ax.set_xlabel('Observed ' + pl.D18O_LABEL, fontsize=FONT)
    if ylabels:
        ax.set_ylabel('Reconstructed ' + pl.D18O_LABEL, fontsize=FONT)
    pl.panel_label(ax, label, font=FONT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='E:/Data/d18o_so/')
    p.add_argument('--out', default=os.path.join(ROOT, 'figures', 'review', 'fig9_acc_divergence_na_crossplots.png'))
    a = p.parse_args()
    config.set_data_path(a.data)

    print('Loading ...', flush=True)
    prods = pr.load_products(ALL)
    _, clim = pr.load_climatology()

    na = na_observations(prods, clim)
    print(f'NA-like GISS observations: {len(na)}')
    for name, *_ in SECTORS:
        s = na[na.sector == name]
        if len(s):
            print(f'  {name}: n = {len(s)}; ' + '; '.join(
                f'{k.upper()} RMSE {np.sqrt(np.mean((s[k] - s.d18O) ** 2)):.3f}, '
                f'bias {(s[k] - s.d18O).mean():+.3f}' for k in RECONS))
    sec = acc_section(prods, clim)

    fig = plt.figure(figsize=(15, 19))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 2.25], hspace=0.12,
                             left=0.08, right=0.88, top=0.98, bottom=0.05)
    top = outer[0].subgridspec(1, 3, wspace=0.08)
    bottom = outer[1].subgridspec(2, 1, hspace=0.22)

    allv = np.concatenate([na.d18O.values] + [na[k].values for k in RECONS])
    lims = (np.floor(allv.min() * 10 - 0.3) / 10, np.ceil(allv.max() * 10 + 0.3) / 10)
    # no tick on the upper edge, so the labels of neighbouring panels do not run into each other
    ticks = np.round(np.arange(np.ceil(lims[0] * 10) / 10, lims[1] - 1e-9, 0.1), 1)
    for j, k in enumerate(RECONS):
        ax = fig.add_subplot(top[0, j])
        draw_crossplot(ax, na, k, f'({"abc"[j]}) {prods[k].label}', lims, ticks, ylabels=(j == 0))

    handles = [Line2D([], [], ls='', marker='o', ms=8, color=col, label=name) for name, _, _, col in SECTORS]
    handles += [Line2D([], [], color='0.4', ls='--', lw=1, label='1:1'),
                plt.Rectangle((0, 0), 1, 1, color=PREC_COLOR, alpha=0.35, label=f'± {PRECISION} ‰')]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.885, outer[0].get_position(fig).y0
                                                                    + 0.5 * outer[0].get_position(fig).height),
               fontsize=FONT - 2, frameon=False)

    cmap = plt.get_cmap('viridis', len(LEVELS))
    norm = BoundaryNorm(LEVELS, cmap.N, extend='max')
    ax_d = fig.add_subplot(bottom[0])
    draw_section(ax_d, sec['R'], sec, cmap, norm, '(d)', dots=giss_in_band(clim), na=na)
    ax_e = fig.add_subplot(bottom[1], sharex=ax_d)
    im = draw_section(ax_e, sec['all'], sec, cmap, norm, '(e)')
    plt.setp(ax_d.get_xticklabels(), visible=False)
    xt = np.arange(0, 361, 30)
    ax_e.set_xticks(xt)
    ax_e.set_xticklabels([f'{int((t + LON0 + 180) % 360 - 180)}°' for t in xt])
    ax_e.set_xlabel('Longitude (°E), eastward from Drake Passage', fontsize=FONT)

    p0, p1 = ax_e.get_position(), ax_d.get_position()
    cax = fig.add_axes([0.905, p0.y0, 0.015, p1.y1 - p0.y0])
    cb = fig.colorbar(im, cax=cax, extend='max', ticks=LEVELS)
    cb.set_label('Range between products (‰)', fontsize=FONT)
    cb.ax.set_yticklabels([f'{v:.2f}' if i % 2 == 0 else '' for i, v in enumerate(LEVELS)])
    cb.ax.tick_params(labelsize=FONT - 2)

    fig.savefig(a.out, dpi=200, bbox_inches='tight')
    print('saved', a.out)


if __name__ == '__main__':
    main()
