#!/usr/bin/env python3
"""Render the Faster-style project figures from archived Nanochat reports."""
import json
from pathlib import Path
import statistics
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RED, INK, MUTED = '#8C0000', '#292122', '#79696B'
GRID, CARD = '#EADFE0', '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-nanochat-crimson', 'font.size': 12})


def save(fig, path):
    fig.savefig(path, facecolor='white', metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=10, colors=MUTED, pad=8)
    ax.grid(axis=axis, color=GRID, linewidth=.7)
    ax.set_axisbelow(True)


def benchmark(summary):
    reference = statistics.median([w['seconds_per_step']
                                  for n in ['reference-a', 'reference-b']
                                  for w in summary['timing'][n]]) * 1000
    candidate = statistics.median([w['seconds_per_step']
                                  for w in summary['timing']['candidate']]) * 1000
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-Nanochat, optimized by AI agents', fontsize=18,
             color=RED, weight='bold')
    fig.text(.06, .78, 'D24 · 8 H20 GPUs · 1,048,576 tokens / update · tensorwise FP8',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.21, .24, .43, .45])
    ax.barh([1, 0], [reference, candidate], height=.24, color=['#D7BDBF', RED], zorder=3)
    ax.set_yticks([1, 0], ['Baseline', 'Faster-Nanochat'])
    for y, value in zip([1, 0], [reference, candidate]):
        ax.text(value + 75, y, f'{value:,.1f} ms', va='center', color=INK,
                fontsize=11, weight='bold')
    ax.set(xlim=(0, 4400), ylim=(-.6, 1.6), xticks=[0, 1000, 2000, 3000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xlabel('Time per optimizer update (ms) ↓', color=MUTED, labelpad=12, fontsize=10.5)
    frame(ax, 'x')
    fig.text(.85, .54, f"{summary['speedup']:.4f}×", color=RED, fontsize=29,
             weight='bold', ha='center')
    fig.text(.85, .43, 'update speedup', color=MUTED, fontsize=11, ha='center')
    save(fig, HERE / '01-h20-benchmark.svg')


def history(summary):
    rows = json.loads((HERE / 'history.json').read_text())['rows']
    fig = plt.figure(figsize=(9, 9), facecolor='white')
    fig.text(.08, .96, 'Faster-Nanochat', color=RED, fontsize=11, weight='bold')
    fig.text(.08, .91, 'Agent Optimization History', color=INK, fontsize=22, weight='bold')
    fig.text(.94, .93, f"{summary['speedup']:.4f}×", color=RED, fontsize=25,
             weight='bold', ha='right')
    fig.text(.94, .904, 'final revalidation', color=MUTED, fontsize=10, ha='right')
    fig.text(.08, .855, 'Cumulative full-check results · 8 H20 GPUs · 50 exploration rounds',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.09, .535, .85, .285])
    rounds = [0] + [h['round'] for h in rows]
    values = [1.] + [h['speedup'] for h in rows]
    ax.plot(rounds, values, color=RED, linewidth=2.3, marker='o', markersize=5.5)
    for index in [0, 3, 7]:
        row = rows[index]
        ax.annotate(f"{row['speedup']:.4f}×", (row['round'], row['speedup']),
                    xytext=(0, 11), textcoords='offset points', ha='center',
                    fontsize=10.5, weight='bold', color=RED)
    ax.set(xticks=[0, 11, 16, 22, 28, 33, 43, 50], xlim=(-2, 53),
           ylim=(.997, 1.087), yticks=[1, 1.02, 1.04, 1.06, 1.08])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}×'))
    ax.set_xlabel('Exploration round', color=MUTED, fontsize=10)
    frame(ax)
    titles = ['Extract FP8 scale arithmetic', 'Pack RMS backward rows',
              'Pack row maxima', 'Select native FP8 GEMM',
              'Extract persistent scales', 'Pack MLP operands',
              'Native value-embedding gradients', 'Tune packed MLP tiles']
    for index, (row, label) in enumerate(zip(rows, titles)):
        x = .09 + (index % 2) * .44
        y = .40 - (index // 2) * .088
        fig.add_artist(FancyBboxPatch((x, y), .41, .067,
                                     boxstyle='round,pad=0.004,rounding_size=0.007',
                                     transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(x + .012, y + .035, f"R{row['round']}", color=RED, fontsize=11,
                 weight='bold', va='center')
        fig.text(x + .078, y + .035, textwrap.fill(label, 27), color=INK,
                 fontsize=10.5, va='center', weight='bold')
    fig.text(.09, .075, 'Round43 retained; Round44–50 did not establish further benefit.',
             color=MUTED, fontsize=10)
    fig.text(.09, .045, 'Cumulative checkpoints, not isolated ablations. Final full check: 1.074368586×.',
             color=MUTED, fontsize=9)
    fig.text(.09, .022, 'Target 1.5× not met. Full-schedule training and BPB / CORE remain unrun.',
             color=MUTED, fontsize=9)
    save(fig, HERE / '02-history.svg')


def modulemap():
    fig = plt.figure(figsize=(9, 6.6), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.text(.055, .92, 'What changed in Faster-Nanochat', fontsize=22, weight='bold', color=INK)
    cards = [
        ('FP8 scales & reductions', 'Extract repeated scale arithmetic\nPack independent reduction rows',
         'scalar_scale_split.py · persistent_scale_split.py\nnorm_rowpack.py · max_rowpack.py', 4),
        ('MLP layout & launch', 'Pair BF16 operand transport\nTune packed tiles and warps',
         'mlp_packed_layout.py\nmlp_packed_tiles.py', 2),
        ('Native backward paths', 'Guarded native FP8 GEMM selection\nFP32 value-embedding gradients',
         'lt_autotune.py · lt_autotune.cpp\nnative_value_embedding.py', 3),
        ('Model integration', 'Default-off optimization switches\nPreserve original recipe and fallbacks',
         'fp8.py\ngpt.py', 2),
    ]
    for index, (title, description, names, count) in enumerate(cards):
        x, y = .055 + (index % 2) * .455, .49 - (index // 2) * .39
        ax.add_patch(FancyBboxPatch((x, y), .425, .34,
                                   boxstyle='round,pad=0.009,rounding_size=0.012',
                                   facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x + .425], [y + .34, y + .34], color=RED, linewidth=2)
        ax.text(x + .014, y + .30, f'{index+1:02d}', color=RED, fontsize=11, weight='bold')
        ax.text(x + .41, y + .30, f'{count} files', color=MUTED, fontsize=10, ha='right')
        ax.text(x + .014, y + .252, title, color=INK, fontsize=13, weight='bold')
        ax.text(x + .014, y + .198, description, color=INK, fontsize=10.3,
                va='top', linespacing=1.6)
        ax.text(x + .014, y + .035, names, color=MUTED, fontsize=8.5, linespacing=1.5)
    save(fig, ROOT / 'docs/change-views/module-map.svg')


if __name__ == '__main__':
    summary = json.loads((HERE / 'history.json').read_text())['benchmark']
    benchmark(summary)
    history(summary)
    modulemap()
    print('Rendered three Faster-Nanochat figures from archived measurements; no GPU work.')
