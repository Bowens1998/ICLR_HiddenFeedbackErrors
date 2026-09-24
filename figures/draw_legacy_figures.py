#!/usr/bin/env python3
"""Regenerate included historical Figures 7–9 from exact archived plot values.

The bundled JSON copies the original plotting arrays without rounding, statistical
recalculation, or sample changes. This script writes figures only. The original
source hashes remain in legacy_figure_data.json; legacy scientific reports and
tables are never rewritten. SVG labels remain editable text.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def save(fig, out, name):
    fig.tight_layout(pad=.6)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / (name + '.' + ext), dpi=220)
    plt.close(fig)


def confirmation(data, out):
    rows = data['confirmation']
    assert len(rows) == 5
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    yy = np.arange(5)
    means = np.array([c['mean'] for c in rows])
    ci = np.array([c['interval'] for c in rows])
    for i, c in enumerate(rows):
        assert len(c['model_points']) == 6
        ax.scatter(c['model_points'], i + np.linspace(-.13, .13, 6),
                   s=13, color='#8dabb6', alpha=.8)
    ax.errorbar(means, yy, xerr=np.stack([means-ci[:, 0], ci[:, 1]-means]),
                fmt='o', capsize=3, ms=4, color='#663885')
    ax.axvline(0, color='#555', ls=':', lw=1)
    ax.set_yticks(yy, ['Coordinate teacher\nActual − free',
                      'Physical labels\nActual − free',
                      'Latent\nActual − donor',
                      'Coordinate teacher\nActual − donor',
                      'Physical labels\nActual − donor'])
    ax.tick_params(axis='y', labelsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('Actual − comparator:\nterminal block-position MSE change (pixels²)', fontsize=8.6)
    ax.grid(axis='x', alpha=.15)
    save(fig, out, 'confirmation')


def reversal(data, out):
    means = np.asarray(data['reversal']['means'])
    ci = np.asarray(data['reversal']['intervals'])
    fig, ax = plt.subplots(figsize=(5.5, 2.05))
    ax.axvline(0, color='#666', lw=1, ls=':')
    for i, (m, c, color) in enumerate(zip(means, ci, ['#278679', '#ab6536'])):
        ax.errorbar(m, i, xerr=[[m-c[0]], [c[1]-m]], fmt='o', color=color,
                    capsize=4, ms=5)
    ax.set_yticks([0, 1], ['Observed-history\nprediction', 'Free rollout'])
    ax.invert_yaxis()
    ax.set_ylim(1.6, -.6)
    ax.set_xlim(-3500, 3500)
    ax.set_xticks([-3000, -1500, 0, 1500, 3000])
    ax.grid(axis='x', alpha=.15)
    ax.set_xlabel('Physical labels − Coordinate teacher:\nterminal block-position MSE difference (pixels²)', fontsize=8.6)
    save(fig, out, 'reversal')


def reacher(data, out):
    r = data['reacher']
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for path, label, color, style in [
            ('free', 'Free rollout', '#777777', '-'),
            ('fiber', 'Readout-preserving\ncorrection', '#8b459e', '-'),
            ('reset', 'Full reset', '#278679', '--')]:
        ax.plot(r['horizons'], r['curves'][path], 'o'+style,
                color=color, label=label, ms=3)
    ax.set_xticks(r['horizons'])
    ax.set_xlabel('Primitive-action horizon')
    ax.set_ylabel('Joint-angle MSE (rad²)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.27), ncol=3,
              frameon=False, fontsize=8)
    ax.grid(alpha=.15)
    save(fig, out, 'reacher_transfer')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE/'legacy_figure_data.json')
    p.add_argument('--output-dir', type=Path, default=HERE)
    args = p.parse_args()
    raw = args.data.read_bytes()
    data = json.loads(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    confirmation(data, args.output_dir)
    reversal(data, args.output_dir)
    reacher(data, args.output_dir)
    print(json.dumps({'figures': ['confirmation', 'reversal', 'reacher_transfer'],
                      'data_sha256': hashlib.sha256(raw).hexdigest()}))


if __name__ == '__main__':
    main()
