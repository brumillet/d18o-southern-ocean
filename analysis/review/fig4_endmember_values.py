# -*- coding: utf-8 -*-
"""
Figure 4 redrawn for the reviewers (R1.4, R1.6, R1.11, R3.22).

Optimised end-members of each of the 12 Southern Ocean datasets for TMI, OCIM
and NEMO (one column per model): delta-18O in (a)-(c), absolute salinity in (d)-(f).
  - one colour-blind-safe colour (Okabe-Ito) per end-member and one shape per
    model (TMI circles, OCIM squares, NEMO triangles), as in Figure 5
    (plotting.ENDMEMBER_COLORS, plotting.MODEL_MARKERS),
  - filled symbols: datasets included in the average; open symbols: datasets
    excluded from it (R1.6, R1.11),
  - dashed line: average over the included datasets (the value used for the
    averaged reconstruction, printed at the right of each panel),
  - shading: 95 % interval of that average from the two-stage bootstrap
    (stations within each dataset, then the datasets entering the average,
    resampled with replacement, 1000 draws each; R3.15), unless --no-ci,
  - region names written out in the legend, without acronyms (R1.4),
  - RMSE of the averaged reconstruction against all bottles, with units.

The figure shows the HS, NA and AA end-members. --with-ms adds the MS
end-member (R3.22), with its own exclusion criterion (MS fraction >= 0.2 in at
least one bottle); that version is for checking only and is not committed.

The end-members are re-optimised exactly as in notebooks/optimization_fractions.ipynb
(cell 26 for delta-18O, cell 30 for salinity); exclusions are those of
TracerOptimizer.calculate_exclusion_indices.

Run from the repository root:
    python analysis/review/fig4_endmember_values.py --data E:/Data/d18o_so/
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import lsq_linear

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import usefull_functions as uf  # noqa: E402
import plotting as pl  # noqa: E402
from tracer_optimization import TracerOptimizer  # noqa: E402

FONT = 14
MODELS = ['TMI', 'OCIM', 'NEMO']
DYES = ['dyeLL', 'dyeMS', 'dyeNP', 'dyeHS', 'dyeNA', 'dyeAA']
# Colour identifies the end-member and shape the model, as in Figure 5:
# each panel is one model, so all its symbols share that model's shape.
STYLE = {d: dict(name=pl.ENDMEMBER_NAMES[d[3:]], color=pl.ENDMEMBER_COLORS[d[3:]])
         for d in ['dyeMS', 'dyeHS', 'dyeNA', 'dyeAA']}
MS = 55              # marker area (pt^2)
N_BOOT = 1000
SEED = 1

# Optimisation set-up per tracer (notebook cells 26 and 30). LL and NP are fixed
# through near-equal bounds; the other end-members are bounded.
TRACERS = {
    'd18O': dict(
        x0=[0.45, -0.3, -0.35, -0.2, 0.3, -0.4],
        lb=[0.4499, -1.0, -0.3501, -1.0, -1.0, -1.0],
        ub=[0.4501, 1.0, -0.3499, 1.0, 1.0, 1.0],
        ylim=(-1.05, 1.05), yticks=np.round(np.arange(-1, 1.01, 0.2), 1),
        ylabel='End-member ' + pl.D18O_LABEL, unit='‰', fmt='+.2f', gap=0.085),
    'absolute_salinity': dict(
        x0=[35.5, 34.6, 33.5, 34.8, 34.7, 34.4],
        lb=[35.4999, 33.5, 33.4999, 33.5, 33.5, 33.5],
        ub=[35.5001, 35.5, 33.5001, 35.5, 35.5, 35.5],
        ylim=(33.4, 35.6), yticks=np.round(np.arange(33.5, 35.51, 0.25), 2),
        ylabel='End-member absolute salinity\n(g kg$^{-1}$)', unit='g kg$^{-1}$',
        fmt='.2f', gap=0.09),
}


def optimise(data_path, tracer):
    """Same set-up as cells 26 (delta-18O) and 30 (salinity) of the notebook."""
    t = TRACERS[tracer]
    optimizer = TracerOptimizer(data_path)
    res = optimizer.run_complete_optimization(
        tracer_file='giss_d18o_modified.h5', tracer_column=tracer,
        initial_values=np.array(t['x0']), bounds=(np.array(t['lb']), np.array(t['ub'])),
        min_depth=150.0, gamma_range=(27.0, 28.6), latitude_threshold=-40.0)
    return optimizer, res


def bootstrap_ci(optimizer, datasets, tracer, dyes, rng):
    """
    95 % interval of each averaged end-member: resample the stations of each
    dataset and refit (1000 draws), then resample the included datasets, take one
    station-resampled fit of each and average (1000 draws); as
    analysis/review/endmember_bootstrap.py. Returns {(model, dye): (low, high)}.
    """
    lb, ub = np.array(TRACERS[tracer]['lb']), np.array(TRACERS[tracer]['ub'])

    def fit(F, y):
        return lsq_linear(F.T, y, bounds=(lb, ub), method='bvls').x

    ci = {}
    for m in MODELS:
        boot = {}
        for i, df in enumerate(datasets):
            F = np.array([df[f'{d}_{m}'].values for d in DYES])
            y = df[tracer].values
            ok = ~np.isnan(F).any(axis=0) & np.isfinite(y)
            F, y = F[:, ok], y[ok]
            st = df[['Latitude', 'Longitude']].round(2).astype(str).agg('_'.join, axis=1).values[ok]
            groups = [np.where(st == u)[0] for u in np.unique(st)]
            draws = np.empty((N_BOOT, 6))
            for b in range(N_BOOT):
                rows = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
                draws[b] = fit(F[:, rows], y[rows])
            boot[i] = draws
        for d in dyes:
            j = DYES.index(d)
            excl = set(optimizer.get_exclusion_indices(m, d))
            inc = [i for i in range(len(datasets)) if i not in excl]
            means = np.empty(N_BOOT)
            for b in range(N_BOOT):
                pick = rng.choice(inc, len(inc))
                means[b] = np.mean([boot[i][rng.integers(N_BOOT), j] for i in pick])
            ci[m, d] = tuple(np.percentile(means, [2.5, 97.5]))
    return ci


def spread_labels(values, min_gap):
    """Shift label positions apart (keeping their order) so they do not overlap."""
    order = np.argsort(values)
    pos = np.asarray(values, float)[order].copy()
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_gap)
    shift = (np.mean(np.asarray(values)[order]) - np.mean(pos))   # re-centre the group
    out = np.empty_like(pos)
    out[order] = pos + shift
    return out


def plot_row(axes, row, tracer, optimizer, res, dyes, ci):
    t = TRACERS[tracer]
    em = optimizer.end_member_values[tracer]
    means = optimizer.mean_end_members[tracer]
    n = len(res['datasets'])
    x = np.arange(n)
    offs = dict(zip(dyes, np.linspace(-0.22, 0.22, len(dyes))))   # side by side within a dataset
    for j, (ax, m) in enumerate(zip(axes, MODELS)):
        for d in dyes:
            s, k = STYLE[d], DYES.index(d)
            excl = set(optimizer.get_exclusion_indices(m, d))
            inc = np.array([i not in excl for i in range(n)])
            if ci is not None:
                ax.axhspan(*ci[m, d], color=s['color'], alpha=0.15, lw=0, zorder=0)
            ax.axhline(means[m][d], color=s['color'], ls='--', lw=1.4, zorder=1)
            ax.scatter(x[inc] + offs[d], em[m][k, inc], s=MS, marker=pl.MODEL_MARKERS[m],
                       c=s['color'], edgecolors=s['color'], linewidths=1.2, zorder=3)
            ax.scatter(x[~inc] + offs[d], em[m][k, ~inc], s=MS, marker=pl.MODEL_MARKERS[m],
                       facecolors='white', edgecolors=s['color'], linewidths=1.4, zorder=3)

        # averaged values at the right edge, pushed apart where they would overlap
        vals = [means[m][d] for d in dyes]
        for d, yl in zip(dyes, spread_labels(vals, t['gap'])):
            ax.text(n - 0.35, yl, format(means[m][d], t['fmt']), color=STYLE[d]['color'],
                    fontsize=FONT - 1, va='center', ha='left', fontweight='bold')

        rmse = uf.rmse_model_recons(res['tracer_data'], tracer, m, optimizer.mean_end_members)
        # above the axes: points at the bounds would hide labels inside
        ax.text(0.0, 1.02, uf.panels_letter_parenthesis[3 * row + j], transform=ax.transAxes,
                fontsize=FONT, fontweight='bold', ha='left', va='bottom')
        ax.text(0.09, 1.02, m, transform=ax.transAxes, fontsize=FONT, ha='left', va='bottom')
        ax.text(1.0, 1.02, f'RMSE = {rmse:.2f} {t["unit"]}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=FONT - 1)

        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xticks(x)
        ax.set_ylim(*t['ylim'])
        ax.set_yticks(t['yticks'])
        ax.tick_params(labelsize=FONT - 1)
        ax.grid(axis='x', color='0.93', lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        if tracer == 'd18O':
            ax.axhline(0, color='0.8', lw=0.8, zorder=0)
    axes[0].set_ylabel(t['ylabel'], fontsize=FONT)


def plot(results, dyes, out):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), sharex=True)
    fig.subplots_adjust(wspace=0.25, hspace=0.16)
    for row, tracer in enumerate(['d18O', 'absolute_salinity']):
        optimizer, res, ci = results[tracer]
        plot_row(axes[row], row, tracer, optimizer, res, dyes, ci)
        for ax in axes[row, 1:]:
            ax.tick_params(labelleft=False)
    n = len(results['d18O'][1]['datasets'])
    for ax in axes[1]:
        ax.set_xticklabels([str(i + 1) for i in range(n)])
        ax.set_xlabel('Dataset', fontsize=FONT)

    # legend: end-members by colour (patches, so no shape is implied), then the
    # filled/open convention shown with the three model shapes
    def shapes(**kw):
        return tuple(Line2D([], [], ls='', marker=pl.MODEL_MARKERS[m], ms=8, **kw) for m in MODELS)

    h = [Patch(color=STYLE[d]['color']) for d in dyes]
    lab = [STYLE[d]['name'] for d in dyes]
    h += [shapes(color='0.3'), shapes(mfc='white', mec='0.3', mew=1.4),
          Line2D([], [], color='0.3', ls='--', lw=1.4)]
    lab += ['Included in the average', 'Excluded from the average', 'Average (value at right)']
    if results['d18O'][2] is not None:
        h.append(Patch(color='0.3', alpha=0.2, lw=0))
        lab.append('Bootstrap 95% interval of the average')
    fig.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, 0.04),
               ncol=4 if len(dyes) > 3 else 3, fontsize=FONT - 2, frameon=False,
               columnspacing=1.5, handletextpad=0.5, handlelength=3.2,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})

    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('saved', out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='E:/Data/d18o_so/')
    p.add_argument('--with-ms', action='store_true', help='add the MS end-member (not committed)')
    p.add_argument('--no-ci', action='store_true', help='skip the bootstrap shading')
    a = p.parse_args()

    dyes = (['dyeMS'] if a.with_ms else []) + ['dyeHS', 'dyeNA', 'dyeAA']
    rng = np.random.default_rng(SEED)
    results = {}
    for tracer in ['d18O', 'absolute_salinity']:
        optimizer, res = optimise(a.data, tracer)
        ci = None if a.no_ci else bootstrap_ci(optimizer, res['datasets'], tracer, dyes, rng)
        results[tracer] = (optimizer, res, ci)
        for m in MODELS:
            for d in dyes:
                excl = sorted(i + 1 for i in optimizer.get_exclusion_indices(m, d))
                lo, hi = ci[m, d] if ci else (np.nan, np.nan)
                print(f'{tracer:17s} {m:5s} {d[3:]:3s} mean '
                      f'{optimizer.mean_end_members[tracer][m][d]:+.3f} '
                      f'95% [{lo:+.3f}, {hi:+.3f}]; excluded datasets {excl}')

    name = ('d18o_absolute_salinity_values_EM_12_datasets_3models_review'
            + ('_MS' if a.with_ms else '') + '.png')
    plot(results, dyes, os.path.join(ROOT, 'figures', 'optimization', name))


if __name__ == '__main__':
    main()
