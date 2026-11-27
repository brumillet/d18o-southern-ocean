# -*- coding: utf-8 -*-
"""
Figure 5: averaged HS, NA and AA end-members in absolute salinity - delta-18O
space, for TMI, OCIM and NEMO (colour = end-member, shape = model).

Run from the repository root:
    python scripts/fig5_endmembers_salinity_d18o.py --data E:/Data/d18o_so/
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import plotting as pl
from fig4_endmember_values import ROOT, MODELS, optimise

FONT = 14
SHOW = ['HS', 'NA', 'AA']
# grouping ellipses, size and orientation drawn by hand
ELLIPSE = {'HS': dict(angle=-45, ratio=2 / 3, offset=(0, 0)),
           'NA': dict(angle=-40, ratio=1 / 3, offset=(0.04, 0.05)),
           'AA': dict(angle=-25, ratio=2 / 3, offset=(0, 0))}
# name position (g kg-1, permil) and alignment
LABEL = {'HS': dict(xy=(34.03, -0.02), ha='left', text='High-latitude\nSouthern Ocean'),
         'NA': dict(xy=(34.93, 0.30), ha='right', text='North Atlantic'),
         'AA': dict(xy=(34.66, -0.36), ha='right', text='Antarctic\nmargin')}


def plot(d18o, sal, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    for em in SHOW:
        dye, col = 'dye' + em, pl.ENDMEMBER_COLORS[em]
        s = np.array([sal[m][dye] for m in MODELS])
        d = np.array([d18o[m][dye] for m in MODELS])
        for m, si, di in zip(MODELS, s, d):
            ax.scatter(si, di, marker=pl.MODEL_MARKERS[m], s=100, color=col, zorder=3)
        e = ELLIPSE[em]
        ax.add_patch(Ellipse(xy=(s.mean() - e['offset'][0], d.mean() - e['offset'][1]),
                             width=5 * s.std() * e['ratio'], height=5 * d.std(),
                             angle=e['angle'], facecolor='none', edgecolor=col, alpha=0.7,
                             lw=1.5, ls='--'))
        lab = LABEL[em]
        ax.text(*lab['xy'], lab['text'], ha=lab['ha'], va='center', fontsize=FONT,
                color=col, fontweight='bold', linespacing=1.0)

    handles = [Line2D([], [], ls='', marker=pl.MODEL_MARKERS[m], ms=9, color='0.2', label=m)
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
    d18o = optimise(a.data, 'd18O')[0].mean_end_members['d18O']
    sal = optimise(a.data, 'absolute_salinity')[0].mean_end_members['absolute_salinity']
    plot(d18o, sal, os.path.join(ROOT, 'figures', 'optimization',
                                 'absolute_salinity_d18o_endmembers.png'))


if __name__ == '__main__':
    main()
