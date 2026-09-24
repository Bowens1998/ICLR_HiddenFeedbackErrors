"""Draw accepted prospective results only; no fitting or statistical inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def axis_style(ax):
    ax.axvline(0, color='#718096', lw=.8, ls=(0, (3, 3)))
    ax.grid(axis='x', color='#D7E0E8', lw=.6)
    ax.set_axisbelow(True)
    for key in ('top', 'right', 'left'): ax.spines[key].set_visible(False)
    ax.spines['bottom'].set_color('#D7E0E8')
    ax.tick_params(axis='y', length=0)


def point(ax, row, y, color, marker='o'):
    mean, lo, hi = row['estimate'], *row['interval']
    assert lo <= mean <= hi
    ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], color=color, marker=marker,
                capsize=3, ms=5, lw=1.4)


def draw(data, output):
    assert data['status'] == 'ALL_THREE_PROSPECTIVE_RESULTS_ACCEPTED'
    assert len(data['s1']) == 4 and len(data['s3']) == 2
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'text.color': '#203247', 'axes.labelcolor': '#203247',
        'xtick.color': '#596B7C', 'ytick.color': '#203247',
        'svg.fonttype': 'none', 'svg.hashsalt': 'prospective_results', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.5, 3.15), facecolor='white')
    a = fig.add_axes([.18, .32, .295, .46])
    b = fig.add_axes([.735, .70, .24, .12])
    c = fig.add_axes([.735, .22, .24, .23])
    teal, orange = '#087F83', '#C27A35'
    for j, objective in enumerate(('decoded_teacher', 'physical_labels')):
        for k, contrast in enumerate(('U_AC', 'S_AC')):
            row = next(r for r in data['s1'] if r['id'] == objective + '/' + contrast)
            point(a, row, 1-j + (.12 if k == 0 else -.12), teal if k == 0 else orange, 'o' if k == 0 else 's')
    a.set_yticks([1, 0], ['Coordinate\nteacher', 'Physical\nlabels'])
    a.set_ylim(-.55, 1.55); a.set_xlabel('Terminal error decrease (pixels²)', labelpad=7, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=teal, marker='o', label='Free − actual'),
        Line2D([], [], color=orange, marker='s', label='Donor − actual')], loc='lower left',
        bbox_to_anchor=(.025, .085), ncol=2, frameon=False, fontsize=9, columnspacing=1,
        borderaxespad=0)
    diagnosis = dict(estimate=data['s2']['estimate']/1e6,
                     interval=[v/1e6 for v in data['s2']['interval']])
    point(b, diagnosis, 0, teal)
    b.set_yticks([0], ['Baseline −\naugmented']); b.set_ylim(-.5, .5)
    b.set_xlabel('Response-prediction MSE decrease\n(10⁶ pixels⁴)', fontsize=8.5, labelpad=5)
    b.xaxis.set_major_locator(MaxNLocator(nbins=3))
    for i, row in enumerate(data['s3']): point(c, row, 1-i, teal)
    c.set_yticks([1, 0], ['T1 actual − free', 'T1 actual − donor'])
    c.set_ylim(-.5, 1.5); c.set_xlabel('Selected physical cost change', fontsize=9, labelpad=6)
    for ax in (a, b, c): axis_style(ax)
    for x, y, letter, title, subtitle in [
        (.025, .955, 'A', 'Reserved evaluator', '256 recipients; 98.75% intervals'),
        (.535, .955, 'B', 'Incremental diagnosis', '512 test recipients; 95% interval'),
        (.535, .535, 'C', 'Matched decision', '512 recipients; 97.5% intervals')]:
        fig.text(x, y, letter, color=teal, weight='bold', fontsize=11, va='top')
        fig.text(x+.03, y, title, weight='bold', fontsize=11, va='top')
        fig.text(x+.03, y-.066, subtitle, color='#596B7C', fontsize=9, va='top')
    fig.text(.025, .035, 'A/B: positive favors actual guidance or the augmented diagnostic.  C: negative favors actual guidance.',
             fontsize=8, color='#596B7C')
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 200} if extension == 'png' else {}
        if extension == 'pdf': kwargs['metadata'] = {'Title': 'Prospective tests of feedback usefulness', 'Author': ''}
        fig.savefig(output / ('prospective_results.' + extension), **kwargs)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, default=Path(__file__).with_name('prospective_results.json'))
    p.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    a = p.parse_args(); draw(json.loads(a.data.read_text()), a.output_dir)
