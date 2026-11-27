# -*- coding: utf-8 -*-
"""
Paired differences between the TMI, OCIM and NEMO delta-18O reconstructions
(averaged end-members) at the same bottles, per Southern Ocean dataset.

For each dataset and pair A - B: mean and RMSE of A - B over the bottles.
Summary rows: mean over the datasets, and RMSE over the datasets of the mean
differences. The RMSE of each reconstruction against the observations is
given for reference. Values in permil.

Run from the repository root:
    python scripts/pairwise_reconstruction_differences.py --data E:/Data/d18o_so/
    python scripts/pairwise_reconstruction_differences.py --data E:/Data/d18o_so/ --min-depth 1500
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from fig4_endmember_values import MODELS, optimise

PAIRS = list(itertools.combinations(MODELS, 2))


def paired_differences(datasets, min_depth=0.0):
    rows = []
    for i, df in enumerate(datasets):
        df = df[df['Depth'] >= min_depth]
        if df.empty:
            continue
        rec = {m: df[f'd18O_rcst_mean_{m}'].values for m in MODELS}
        row = {'dataset': i + 1, 'n_bottles': len(df)}
        for a, b in PAIRS:
            d = rec[a] - rec[b]
            row[f'{a}-{b} mean'] = d.mean()
            row[f'{a}-{b} RMSE'] = np.sqrt(np.mean(d ** 2))
        for m in MODELS:
            row[f'RMSE {m}-obs'] = np.sqrt(np.mean((rec[m] - df['d18O'].values) ** 2))
        rows.append(row)
    t = pd.DataFrame(rows)

    cols = t.columns[2:]
    mean_row = {'dataset': 'mean', 'n_bottles': t['n_bottles'].sum(),
                **{c: t[c].mean() for c in cols}}
    rmse_row = {'dataset': 'RMSE', **{c: np.sqrt(np.mean(t[c] ** 2))
                                      for c in cols if c.endswith(' mean')}}
    return pd.concat([t, pd.DataFrame([mean_row, rmse_row])], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='E:/Data/d18o_so/')
    p.add_argument('--min-depth', type=float, default=0.0, help='m')
    p.add_argument('--csv', help='optional output file')
    a = p.parse_args()

    t = paired_differences(optimise(a.data, 'd18O')[1]['datasets'], a.min_depth)
    pd.set_option('display.width', 250)
    print(t.round(3).to_string(index=False))
    if a.csv:
        t.round(4).to_csv(a.csv, index=False)


if __name__ == '__main__':
    main()
