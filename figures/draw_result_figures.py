#!/usr/bin/env python3
"""Regenerate Figures 2–4 from the bundled, exact accepted plotting values.

Usage: python draw_result_figures.py [--output-dir DIR]
Requires Python 3, NumPy, and Matplotlib. No experiment or table is changed.
SVG labels remain live text; PDF output contains vector lines and markers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
INK = '#203247'
MUTED = '#596B7C'
GRAY = '#718096'
TEAL = '#087F83'
ORANGE = '#C27A35'
PALE = '#F3F6F9'
LINE = '#D7E0E8'


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.3,
        'axes.labelsize': 10.3, 'axes.titlesize': 11.1,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': MUTED, 'ytick.color': INK,
        'axes.edgecolor': LINE, 'axes.linewidth': .8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })


def style(ax, zero=True):
    ax.grid(axis='x', color=LINE, linewidth=.55, alpha=.72)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.tick_params(axis='x', length=3, width=.6, labelsize=9.8)
    if zero:
        ax.axvline(0, color=GRAY, linewidth=.9, linestyle=(0, (3, 3)), zorder=1)


def heading(fig, x, y, letter, title, subtitle=None):
    fig.text(x, y, letter, fontsize=11.1, fontweight='bold', va='top', color=TEAL)
    fig.text(x+.026, y, title, fontsize=11.1, fontweight='bold', va='top')
    if subtitle:
        fig.text(x+.026, y-.062, subtitle, fontsize=9.6, color=MUTED, va='top')


def save(fig, output, name):
    for extension in ['pdf', 'svg', 'png']:
        kwargs = {'metadata': {'Creator': 'Scientific vector figure source'}} if extension == 'pdf' else {}
        fig.savefig(output / f'{name}.{extension}', dpi=220, **kwargs)
    plt.close(fig)


def forest(ax, rows, labels, positions=None):
    positions = np.arange(len(rows))[::-1] if positions is None else positions
    for y, row in zip(positions, rows):
        values = row['model_points']
        for j, point in enumerate(values):
            ax.scatter(point['paired_change'], y+(j-(len(values)-1)/2)*.027,
                       s=15, color='#A5B3C1', edgecolors='white', linewidth=.3, zorder=2)
        v = row['estimate']; lo, hi = row['interval']
        ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt='o', markersize=5.5,
                    capsize=2.7, linewidth=1.55, color=TEAL, zorder=3)
    ax.set_yticks(positions, labels)
    style(ax)


def confirmation(data, output):
    rows = data['confirmation']
    assert len(rows) == 9
    assert [len(row['model_points']) for row in rows] == [6]*5+[2]*2+[0]*2
    fig = plt.figure(figsize=(7.5, 3.23))
    ax = fig.add_axes([.292, .20, .276, .61])
    dual = fig.add_axes([.748, .635, .222, .172])
    native = fig.add_axes([.748, .20, .222, .172])
    heading(fig, .014, .97, 'A', 'Single pose readout', '6 groups · 6-direction matching family')
    heading(fig, .603, .97, 'B', 'Dual pose readouts', '2 groups · 12-direction matching family')
    heading(fig, .603, .535, 'C', 'Native DINO-WM', '2-direction matching family')
    forest(ax, rows[:5], ['Coordinate teacher:\nactual − free', 'Physical labels:\nactual − free',
                         'Latent: actual − donor', 'Coordinate teacher:\nactual − donor',
                         'Physical labels:\nactual − donor'])
    ax.set_ylim(-.45, 4.45)
    ax.axhline(2.5, color=LINE, linewidth=.65)
    ax.set_xlim(-1320, 100); ax.set_xticks([-1200, -800, -400, 0])
    forest(dual, rows[5:7], ['Coordinate\nteacher', 'Physical labels'])
    dual.set_ylim(-.48, 1.48)
    dual.set_xlim(-1320, 100); dual.set_xticks([-1200, -600, 0])
    dual.text(-.02, 1.20, 'Actual − free', transform=dual.transAxes, ha='left',
              color=MUTED, fontsize=9.0)
    forest(native, rows[7:9], ['Actual − free', 'Actual − donor'])
    native.set_ylim(-.48, 1.48)
    native.set_xlim(-5100, 300); native.set_xticks([-4000, -2000, 0])
    native.text(1, -.65, 'Separate bank; native scale differs', transform=native.transAxes,
                ha='right', color=MUTED, fontsize=8.2)
    fig.text(.294, .117, 'Terminal block-position\nMSE change (pixels²)', ha='left', va='top', fontsize=9.6)
    fig.text(.018, .014, 'Negative favors actual guidance', fontsize=9.1, color=MUTED)
    handles = [Line2D([], [], color=TEAL, marker='o', markersize=5, linewidth=1.5,
                      label='Mean + 99.444% interval'),
               Line2D([], [], color='#A5B3C1', marker='o', markersize=4, linewidth=0,
                      label='Fixed model group')]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(.985, -.02),
               ncol=2, frameon=False, fontsize=8.8, handlelength=1.6,
               columnspacing=1.0)
    save(fig, output, 'confirmation_strengthened')


def motion(data, output):
    assert len(data['motion']['primary']) == len(data['motion']['secondary']) == 6
    fig = plt.figure(figsize=(7.5, 3.13))
    left = fig.add_axes([.235, .185, .284, .60])
    right = fig.add_axes([.750, .185, .231, .60])
    heading(fig, .015, .98, 'A', 'Source benefit and attenuation',
            r'$\Delta = \mathrm{donor} - \mathrm{actual}$')
    heading(fig, .561, .98, 'B', 'Benefit over free rollout',
            r'$\mathrm{free} - \mathrm{actual}$')
    for ax, family, labels in [(left, 'primary',
                               [r'$\Delta_{\mathrm{pose}}$',
                                r'$\Delta_{\mathrm{pose}}-\Delta_{\mathrm{joint}}$',
                                r'$\Delta_{\mathrm{rotated}}-\Delta_{\mathrm{joint}}$']),
                              (right, 'secondary', ['Pose-only', 'Pose + velocity',
                                                     'Pose + rotated\nvelocity'])]:
        for group, color, marker in [(0, TEAL, 'o'), (1, ORANGE, 's')]:
            rows = [r for r in data['motion'][family] if r['group'] == group]
            for i, row in enumerate(rows):
                v = row['estimate']; lo, hi = row['interval']
                ax.errorbar(v, 2-i+(.10 if group == 0 else -.10),
                            xerr=[[v-lo], [hi-v]], fmt=marker, color=color,
                            markersize=5.0, capsize=2.5, linewidth=1.45, zorder=3)
        ax.set_yticks([2, 1, 0], labels)
        ax.set_ylim(-.4, 2.4)
        style(ax)
    left.set_xlim(-205, 830); left.set_xticks([0, 400, 800])
    right.set_xlim(-40, 1350); right.set_xticks([0, 400, 800, 1200])
    fig.text(.60, .075, 'Terminal block-position MSE difference (pixels²)', ha='center', fontsize=10.2)
    handles = [Line2D([], [], color=TEAL, marker='o', linewidth=1.3, markersize=5, label='Transformer'),
               Line2D([], [], color=ORANGE, marker='s', linewidth=1.3, markersize=5, label='GRU')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.014, -.025),
               ncol=2, frameon=False, fontsize=9.4, handlelength=1.6, columnspacing=1.6)
    fig.text(.984, .016, '99.1667% intervals within each family', fontsize=9.0,
             color=MUTED, ha='right')
    save(fig, output, 'motion_constraint_results')


def training(data, output):
    rows = data['training']
    assert len(rows) == 6
    fig = plt.figure(figsize=(7.5, 2.88))
    axes = [fig.add_axes([.11, .415, .392, .44]),
            fig.add_axes([.58, .415, .392, .44])]
    colors = [GRAY, TEAL, ORANGE]; markers = ['s', 'o', '^']
    for ax, objective, title, letter in zip(axes,
            ['decoded_teacher', 'physical_labels'], ['Coordinate teacher', 'Physical labels'], ['A', 'B']):
        values = [r for r in rows if r['objective'] == objective]
        assert [r['condition'] for r in values] == ['T0', 'T1', 'T2']
        means_x = [r['free_mean'] for r in values]
        means_y = [r['G_mean'] for r in values]
        for pool in range(3):
            xx = [r['pool_free'][pool] for r in values]
            yy = [r['pool_G'][pool] for r in values]
            ax.plot(xx, yy, color='#BDC8D3', linewidth=.7, zorder=1)
        ax.plot(means_x, means_y, color=MUTED, linewidth=1.15, zorder=2)
        for k, (r, color, marker) in enumerate(zip(values, colors, markers)):
            ax.scatter(r['pool_free'], r['pool_G'], color=color, marker=marker,
                       s=20, alpha=.50, linewidth=.5, edgecolors='white', zorder=3)
            ax.scatter(r['free_mean'], r['G_mean'], color=color, marker=marker,
                       s=68, linewidth=.65, edgecolors='white', zorder=4)
            if k == 0: offset = (8, 4)
            elif objective == 'decoded_teacher': offset = (8, -13) if k == 1 else (8, 6)
            else: offset = (8, 5) if k == 1 else (8, -14)
            ax.annotate(r['condition'], (r['free_mean'], r['G_mean']), xytext=offset,
                        textcoords='offset points', fontsize=10.7, color=color,
                        fontweight='bold', zorder=5)
        ax.set_title(f'{letter}  {title}', loc='left', fontweight='bold', fontsize=11.0, pad=8)
        ax.set_ylim(225, 760)
        ax.set_xlim(4400, 9000); ax.set_xticks([5000, 6000, 7000, 8000])
        ax.set_yticks([300, 450, 600, 750])
        ax.grid(color=LINE, linewidth=.55, alpha=.72); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='both', length=3, width=.6, labelsize=9.8)
        ax.set_xlabel('Free-rollout terminal\nblock-position MSE (pixels²)', labelpad=6)
    axes[0].set_ylabel(r'Remaining correctability, $G$'+'\n(pixels²)', fontsize=10.2, labelpad=8)
    axes[1].set_yticklabels([])
    handles = [Line2D([], [], color=c, marker=m, linestyle='none', markersize=5.3, label=label)
               for c, m, label in zip(colors, markers,
               ['T0  Observed-history\ntraining', 'T1  Recursive\ntraining',
                'T2  Recursive training\n+ latent anchor'])]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .030),
               ncol=3, frameon=False, fontsize=8.6, handletextpad=.3, columnspacing=1.4)
    fig.text(.54, .014, 'Large points: mean across pools · Small points: each pool · Lines: matched conditions',
             color=MUTED, fontsize=8.8, ha='center')
    save(fig, output, 'training_feedback_joint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE/'result_figure_data.json')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.data.read_bytes(); data = json.loads(raw)
    setup()
    confirmation(data, args.output_dir)
    motion(data, args.output_dir)
    training(data, args.output_dir)
    print(json.dumps({'figures': ['confirmation_strengthened', 'motion_constraint_results',
                                 'training_feedback_joint'],
                      'data_sha256': hashlib.sha256(raw).hexdigest(),
                      'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
