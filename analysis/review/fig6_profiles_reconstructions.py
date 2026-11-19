# -*- coding: utf-8 -*-
"""
Figure 6 redrawn for the reviewers (R1.6 and R3.26).

Observed and reconstructed delta-18O against neutral density for the 12
Southern Ocean datasets, with OCIM (or TMI/NEMO with --model):
  - observations in black,
  - reconstruction with the end-members optimised for each dataset in cyan,
  - reconstruction with the averaged end-members in magenta,
all as small dots (--style dots). With --style bands, each series is instead
averaged in neutral-density bins (0.1 kg m-3 above 27.8, 0.05 kg m-3 below):
a line for the bin mean and a band for +-1 sigma of the bottles in the bin, the
same measure of spread for the observations and both reconstructions.
The map above keeps the dataset colours as the key that
links each panel to its location (R1.5), with the stations as crosses.

The end-members are re-optimised exactly as in notebooks/optimization_fractions.ipynb
(cells 26 and 40-44).

Run from the repository root:
    python analysis/review/fig6_profiles_reconstructions.py --data E:/Data/d18o_so/
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import usefull_functions as uf  # noqa: E402
import plotting as pl  # noqa: E402
import custom_density_scale  # noqa: E402,F401  (registers 'custom_scale')
from tracer_optimization import TracerOptimizer  # noqa: E402

FONT = 14
OBS_COLOR = 'black'
IND_COLOR = '#00b8d4'   # cyan, slightly darkened so the RMSE text stays legible
AVG_COLOR = '#d400d4'   # magenta
XAVG_COLOR = 'black'    # averaged end-members in the crossplot (no observation series there)
DOT = 22                # marker area (pt^2) of the profile dots
DOT_ALPHA = 0.55        # transparent enough to see where the three series overlap
BAND_ALPHA = {OBS_COLOR: 0.25, IND_COLOR: 0.25, AVG_COLOR: 0.18}


def optimise(data_path):
    """Same set-up as cell 26 of optimization_fractions.ipynb."""
    optimizer = TracerOptimizer(data_path)
    lower, upper = np.full(6, -1.0), np.full(6, 1.0)
    lower[0], upper[0] = 0.4499, 0.4501    # dyeLL fixed
    lower[2], upper[2] = -0.3501, -0.3499  # dyeNP fixed
    res = optimizer.run_complete_optimization(
        tracer_file='giss_d18o_modified.h5', tracer_column='d18O',
        initial_values=np.array([0.45, -0.3, -0.35, -0.2, 0.3, -0.4]),
        bounds=(lower, upper), min_depth=150.0, gamma_range=(27.0, 28.6),
        latitude_threshold=-40.0)
    return optimizer, res


def rmse(a, b):
    return np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2))


def binned(gamma, values, edges, min_n=1):
    """Mean and standard deviation of `values` in neutral-density bins."""
    gamma, values = np.asarray(gamma, float), np.asarray(values, float)
    idx = np.digitize(gamma, edges) - 1
    mean = np.full(len(edges) - 1, np.nan)
    std = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        v = values[(idx == b) & np.isfinite(values)]
        if v.size >= min_n:
            mean[b], std[b] = v.mean(), v.std()
    return mean, std


# kg m-3; coarser above 27.8 where the bottles are sparse, finer in the deep ocean
GAMMA_EDGES = np.unique(np.round(np.concatenate([np.arange(27.0, 27.8, 0.1),
                                       np.arange(27.8, 28.6 + 1e-9, 0.05)]), 3))
GAMMA_MID = 0.5 * (GAMMA_EDGES[1:] + GAMMA_EDGES[:-1])


def plot(datasets, model, out, style='dots'):
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(4, 4, wspace=0.04, hspace=0.1)
    couleurs = plt.cm.tab20(range(12))

    # --- map: colour key of the datasets -------------------------------------
    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=0)
    gl = ax.gridlines(draw_labels={'bottom': 'x', 'right': 'y', 'left': 'y'},
                      ylocs=[-90 + 15 * i for i in range(5)])
    gl.xlabel_style = {'size': FONT - 1}
    gl.ylabel_style = {'size': FONT - 1}
    ax.set_ylim(-90, -20)
    ax.set_xlim(-180, 180)
    for i in range(12):
        ax.scatter(datasets[i]['Longitude'], datasets[i]['Latitude'], color=couleurs[i],
                   marker='x', transform=ccrs.PlateCarree())
        ax.text(uf.coords_dataset[i][0], uf.coords_dataset[i][1], f'({i + 1})',
                fontsize=FONT, color=couleurs[i])
    ax.text(0.99, 1.02, model, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT + 5, fontweight='bold')

    # --- profiles --------------------------------------------------------------
    for i in range(12):
        df = datasets[i]
        i_row, i_col = i // 4 + 1, i % 4
        xlabels, ylabels = uf.get_xylabels(3, 4, i)
        ax = fig.add_subplot(gs[i_row, i_col])

        ind, avg = df['d18O_rcst_' + model], df['d18O_rcst_mean_' + model]
        if style == 'dots':
            # reconstructions first, observations on top
            ax.scatter(ind, df['gamma_n'], s=DOT, c=IND_COLOR, alpha=DOT_ALPHA, linewidths=0)
            ax.scatter(avg, df['gamma_n'], s=DOT, c=AVG_COLOR, alpha=DOT_ALPHA, linewidths=0)
            sc = ax.scatter(df['d18O'], df['gamma_n'], s=DOT, c=OBS_COLOR, alpha=DOT_ALPHA,
                            linewidths=0)
        else:
            g = df['gamma_n']
            # faint bottles underneath, to show where the data are
            sc = ax.scatter(df['d18O'], g, s=4, c=OBS_COLOR, alpha=0.2, linewidths=0)
            # bin mean and +-1 sigma of the bottles in each bin, for all three series
            for values, col in [(avg, AVG_COLOR), (ind, IND_COLOR), (df['d18O'], OBS_COLOR)]:
                m, sd = binned(g, values, GAMMA_EDGES)
                ax.fill_betweenx(GAMMA_MID, m - sd, m + sd, color=col,
                                 alpha=BAND_ALPHA[col], linewidth=0)
                ax.plot(m, GAMMA_MID, c=col, lw=2)

        # statistics in the bottom-right corner, which is empty in every panel
        box = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1)
        for k, (txt, col) in enumerate([
                (f'RMSE$_{{ind}}$ = {rmse(df["d18O"], ind):.3f} ‰', IND_COLOR),
                (f'RMSE$_{{avg}}$ = {rmse(df["d18O"], avg):.3f} ‰', AVG_COLOR),
                (f'$\\Delta_{{avg}}$ = {np.nanmean(avg - df["d18O"]):+.2f} ‰', AVG_COLOR)]):
            ax.text(0.97, 0.18 - 0.07 * k, txt, transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=FONT - 2, color=col, bbox=box, zorder=5)
        pl.panel_label(ax, f'({i + 1})', font=FONT, letter_color=couleurs[i],
                       letter_weight='normal')

        ax.set_yscale('custom_scale')
        uf.plot_details_axis(ax, sc, cb=False, xlim=(-0.7, 0.5), ylim=(28.6, 27), nbins=20,
                             xticks=[-0.6 + 0.2 * k for k in range(6)],
                             yticks=[27 + 0.2 * k for k in range(9)], font=FONT,
                             xlabels=xlabels, ylabels=ylabels)
        ax.invert_yaxis()
        if ylabels:
            ax.set_yticklabels(['27', '', '27.4', '', '27.8', '', '28.2', '', '28.6'],
                               fontsize=FONT)
            ax.set_ylabel(pl.GAMMA_LABEL, fontsize=FONT)
        if xlabels:
            ax.set_xlabel(pl.D18O_LABEL, fontsize=FONT)

        if xlabels:
            ax.set_xticklabels(['', '-0.4', '', '0.0', '', '0.4'], fontsize=FONT)
        if i == 11:
            bottom = ax.get_position().y0

    if style == 'dots':
        handles = [Line2D([], [], ls='', marker='o', ms=9, color=c, alpha=DOT_ALPHA, label=l)
                   for c, l in
                   [(OBS_COLOR, 'Observations'),
                    (IND_COLOR, 'Individual end-members'),
                    (AVG_COLOR, 'Averaged end-members')]]
    else:
        from matplotlib.patches import Patch
        handles, labels = [], []
        for col, lab in [(OBS_COLOR, 'Observations'),
                         (IND_COLOR, 'Individual end-members'),
                         (AVG_COLOR, 'Averaged end-members')]:
            handles.append((Patch(color=col, alpha=BAND_ALPHA[col], lw=0),
                            Line2D([], [], color=col, lw=2)))
            labels.append(lab)
        fig.legend(handles, labels, fontsize=FONT, ncol=3, loc='upper center',
                   bbox_to_anchor=(0.5, bottom - 0.035), frameon=False)
        fig.text(0.5, bottom - 0.06, 'Lines: mean in each neutral-density bin; '
                 'shading: ±1σ of the bottles in the bin.', ha='center', va='top',
                 fontsize=FONT - 1)
    if style == 'dots':
        fig.legend(handles=handles, fontsize=FONT, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, bottom - 0.035), frameon=False)

    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('saved', out)


def plot_crossplot(datasets, model, out):
    """
    Observed against reconstructed delta-18O, one panel per dataset, with both
    reconstructions (individual end-members in cyan, averaged in magenta) and the
    1:1 line. Same map, colours and dots as the profile figure.
    """
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 4, wspace=0.06, hspace=0.08, height_ratios=[0.85, 1, 1, 1])
    couleurs = plt.cm.tab20(range(12))

    ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=0)
    gl = ax.gridlines(draw_labels={'bottom': 'x', 'right': 'y', 'left': 'y'},
                      ylocs=[-90 + 15 * i for i in range(5)])
    gl.xlabel_style = {'size': FONT - 1}
    gl.ylabel_style = {'size': FONT - 1}
    ax.set_ylim(-90, -20)
    ax.set_xlim(-180, 180)
    for i in range(12):
        ax.scatter(datasets[i]['Longitude'], datasets[i]['Latitude'], color=couleurs[i],
                   marker='x', transform=ccrs.PlateCarree())
        ax.text(uf.coords_dataset[i][0], uf.coords_dataset[i][1], f'({i + 1})',
                fontsize=FONT, color=couleurs[i])
    ax.text(0.99, 1.02, model, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT + 5, fontweight='bold')

    # common square axis range covering every bottle, rounded out to 0.1 permil
    cols = ['d18O', 'd18O_rcst_' + model, 'd18O_rcst_mean_' + model]
    allv = np.concatenate([np.asarray(d[c], float) for d in datasets for c in cols])
    lims = (np.floor(np.nanmin(allv) * 10 - 0.5) / 10, np.ceil(np.nanmax(allv) * 10 + 0.5) / 10)
    ticks = np.round(np.arange(np.ceil(lims[0] * 5) / 5, lims[1] + 1e-9, 0.2), 1)

    for i in range(12):
        df = datasets[i]
        i_row, i_col = i // 4 + 1, i % 4
        xlabels, ylabels = uf.get_xylabels(3, 4, i)
        ax = fig.add_subplot(gs[i_row, i_col])
        obs = df['d18O']
        ind, avg = df['d18O_rcst_' + model], df['d18O_rcst_mean_' + model]

        ax.plot(lims, lims, color='0.5', ls='--', lw=1, zorder=0)
        ax.scatter(obs, avg, s=DOT, c=XAVG_COLOR, alpha=DOT_ALPHA, linewidths=0)
        ax.scatter(obs, ind, s=DOT, c=IND_COLOR, alpha=DOT_ALPHA, linewidths=0)

        box = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1)
        for k, (txt, col) in enumerate([
                (f'RMSE$_{{ind}}$ = {rmse(obs, ind):.3f} ‰', IND_COLOR),
                (f'RMSE$_{{avg}}$ = {rmse(obs, avg):.3f} ‰', XAVG_COLOR),
                (f'$\\Delta_{{avg}}$ = {np.nanmean(avg - obs):+.2f} ‰', XAVG_COLOR)]):
            # top-left corner, above the 1:1 line, is empty in every panel
            ax.text(0.03, 0.87 - 0.07 * k, txt, transform=ax.transAxes, ha='left',
                    va='top', fontsize=FONT - 2, color=col, bbox=box, zorder=5)
        pl.panel_label(ax, f'({i + 1})', font=FONT, letter_color=couleurs[i],
                       letter_weight='normal')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f'{t:.1f}' for t in ticks] if xlabels else [], fontsize=FONT)
        ax.set_yticklabels([f'{t:.1f}' for t in ticks] if ylabels else [], fontsize=FONT)
        if xlabels:
            ax.set_xlabel('Observed ' + pl.D18O_LABEL, fontsize=FONT)
        if ylabels:
            ax.set_ylabel('Reconstructed ' + pl.D18O_LABEL, fontsize=FONT)
        if i == 11:
            bottom = ax.get_position().y0

    handles = [Line2D([], [], ls='', marker='o', ms=9, color=c, alpha=DOT_ALPHA, label=l)
               for c, l in [(IND_COLOR, 'Individual end-members'),
                            (XAVG_COLOR, 'Averaged end-members')]]
    handles.append(Line2D([], [], color='0.5', ls='--', lw=1, label='1:1 line'))
    fig.legend(handles=handles, fontsize=FONT, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, bottom - 0.035), frameon=False)

    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('saved', out, 'axis range', lims)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='E:/Data/d18o_so/')
    p.add_argument('--model', default='OCIM', choices=['OCIM', 'TMI', 'NEMO'])
    p.add_argument('--style', default='both',
                   choices=['dots', 'bands', 'both', 'crossplot', 'all'],
                   help='dots: every bottle; bands: binned means with 1-sigma bands; '
                        'both: dots and bands; crossplot: observed vs reconstructed')
    a = p.parse_args()
    _, res = optimise(a.data)
    styles = {'both': ['dots', 'bands'], 'all': ['dots', 'bands', 'crossplot']}.get(
        a.style, [a.style])
    if 'crossplot' in styles:
        styles.remove('crossplot')
        plot_crossplot(res['datasets'], a.model, os.path.join(
            ROOT, 'figures', 'optimization', f'd18o_crossplot_{a.model}_review.png'))
    for style in styles:
        suffix = '' if style == 'dots' else '_bands'
        out = os.path.join(ROOT, 'figures', 'optimization',
                           f'd18o_profiles_SO_map_{a.model}_rcs_dataset_&mean_review{suffix}.png')
        plot(res['datasets'], a.model, out, style=style)


if __name__ == '__main__':
    main()
