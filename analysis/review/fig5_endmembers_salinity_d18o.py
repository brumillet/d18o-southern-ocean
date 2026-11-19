# -*- coding: utf-8 -*-
"""
Figure 5 redrawn for the reviewers (R1.4, R3.22).

Averaged end-members of the three deep-ventilating surface regions in
absolute salinity - delta-18O space, for TMI, OCIM and NEMO:
  - colour = end-member, the same colours as Figure 4 (plotting.ENDMEMBER_COLORS),
  - marker shape = model,
  - region names written out in full, without acronyms (R1.4),
  - dashed ellipses group the three models of each end-member (as in the
    submitted figure; their size and orientation are drawn by hand).
MS is not shown (R3.22; see the SI version of Figure 4).

Both optimisations are those of notebooks/optimization_fractions.ipynb
(cell 26 for delta-18O, cell 30 for salinity); the plot follows cell 38.

Run from the repository root:
    python analysis/review/fig5_endmembers_salinity_d18o.py --data E:/Data/d18o_so/
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import plotting as pl  # noqa: E402
from tracer_optimization import TracerOptimizer  # noqa: E402

FONT = 14
MODELS = ['TMI', 'OCIM', 'NEMO']
MODEL_MARKERS = pl.MODEL_MARKERS   # same shapes as Figure 4
SHOW = ['HS', 'NA', 'AA']
# hand-drawn grouping ellipses, unchanged from the submitted figure
ELLIPSE = {'HS': dict(angle=-45, ratio=2 / 3, offset=(0, 0)),
           'NA': dict(angle=-40, ratio=1 / 3, offset=(0.04, 0.05)),
           'AA': dict(angle=-25, ratio=2 / 3, offset=(0, 0))}
# name position (absolute salinity in g kg-1, delta-18O in permil) and alignment
LABEL = {'HS': dict(xy=(34.03, -0.02), ha='left', text='High-latitude\nSouthern Ocean'),
         'NA': dict(xy=(34.93, 0.30), ha='right', text='North Atlantic'),
         'AA': dict(xy=(34.66, -0.36), ha='right', text='Antarctic\nmargin')}


def optimise(data_path):
    """delta-18O (cell 26) and absolute salinity (cell 30) end-members."""
    d18o = TracerOptimizer(data_path)
    lo, hi = np.full(6, -1.0), np.full(6, 1.0)
    lo[0], hi[0] = 0.4499, 0.4501      # LL fixed
    lo[2], hi[2] = -0.3501, -0.3499    # NP fixed
    d18o.run_complete_optimization(
        tracer_file='giss_d18o_modified.h5', tracer_column='d18O',
        initial_values=np.array([0.45, -0.3, -0.35, -0.2, 0.3, -0.4]), bounds=(lo, hi),
        min_depth=150.0, gamma_range=(27.0, 28.6), latitude_threshold=-40.0)

    sal = TracerOptimizer(data_path)
    sal.run_complete_optimization(
        tracer_file='giss_d18o_modified.h5', tracer_column='absolute_salinity',
        initial_values=np.array([35.5, 34.6, 33.5, 34.8, 34.7, 34.4]),
        bounds=(np.array([35.4999, 33.5, 33.4999, 33.5, 33.5, 33.5]),
                np.array([35.5001, 35.5, 33.5001, 35.5, 35.5, 35.5])),
        min_depth=150.0, gamma_range=(27.0, 28.6), latitude_threshold=-40.0)
    return d18o.mean_end_members['d18O'], sal.mean_end_members['absolute_salinity']


def plot(d18o, sal, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    for em in SHOW:
        dye, col = 'dye' + em, pl.ENDMEMBER_COLORS[em]
        s = np.array([sal[m][dye] for m in MODELS])
        d = np.array([d18o[m][dye] for m in MODELS])
        for m, si, di in zip(MODELS, s, d):
            ax.scatter(si, di, marker=MODEL_MARKERS[m], s=100, color=col, zorder=3)
            print(f'{em} {m:4s}: S = {si:.3f} g kg-1, d18O = {di:+.3f} permil')
        e = ELLIPSE[em]
        ax.add_patch(Ellipse(xy=(s.mean() - e['offset'][0], d.mean() - e['offset'][1]),
                             width=5 * s.std() * e['ratio'], height=5 * d.std(),
                             angle=e['angle'], facecolor='none', edgecolor=col, alpha=0.7,
                             lw=1.5, ls='--'))
        lab = LABEL[em]
        ax.text(*lab['xy'], lab['text'], ha=lab['ha'], va='center', fontsize=FONT,
                color=col, fontweight='bold', linespacing=1.0)

    handles = [Line2D([], [], ls='', marker=MODEL_MARKERS[m], ms=9, color='0.2', label=m)
               for m in MODELS]
    ax.legend(handles=handles, loc='upper left', title='Models', fontsize=FONT - 2,
              title_fontsize=FONT - 1)

    ax.set_xlabel('Absolute salinity (g kg$^{-1}$)', fontsize=FONT)
    ax.set_ylabel(pl.D18O_LABEL, fontsize=FONT)
    ax.set_xlim(34.0, 35.4)
    ax.set_ylim(-0.64, 0.6)
    ax.set_xticks(np.round(34 + 0.2 * np.arange(8), 1))
    ax.set_yticks(np.round(-0.6 + 0.2 * np.arange(7), 1))
    ax.tick_params(labelsize=FONT - 1)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('saved', out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='E:/Data/d18o_so/')
    a = p.parse_args()
    d18o, sal = optimise(a.data)
    plot(d18o, sal, os.path.join(ROOT, 'figures', 'optimization',
                                 'absolute_salinity_d18o_endmembers_review.png'))


if __name__ == '__main__':
    main()
