"""Figures for the README from results/binance_analysis.json (real Binance data, test period only).

    $PORTFOLIO_VENV/bin/python scripts/plot_binance.py      # any Python with matplotlib

Writes docs/figures/rolling_ic.png and docs/figures/cumulative_net.png. matplotlib is needed
here only; the library and its tests use the standard library.
"""
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 60
# Categorical slots in fixed order; the pick is drawn heavier so it reads first.
STYLE = {'gp_pick': ('GP pick', '#2a78d6', 2.4),
         'random_search_pick': ('random-search pick (equal budget)', '#eb6834', 1.4),
         'range_10d': ('simple control: 10-day mean range', '#1baf7a', 1.4),
         'momentum_20d': ('momentum_20d', '#eda100', 1.2),
         'reversal_1d': ('reversal_1d', '#e87ba4', 1.2)}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def rolling(xs, n):
    out = []
    for i in range(len(xs)):
        w = [x for x in xs[max(0, i - n + 1):i + 1] if x is not None]
        out.append(sum(w) / len(w) if i >= n - 1 and w else None)
    return out


def cumulative(xs):
    total, out = 0.0, []
    for x in xs:
        total += x
        out.append(total)
    return out


def main():
    a = json.loads((ROOT / 'results' / 'binance_analysis.json').read_text())
    days = [date.fromisoformat(d) for d in a['test_dates']]
    fee, series = a['fee_bps_per_side'], a['series']
    names = [n for n in STYLE if n in series]
    out = ROOT / 'docs' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    note = (f'Binance spot daily, 34 surviving coins (survivorship-biased), test 2025-01-01 to 2026-08-31 '
            f'({len(days)} days). Source: results/binance_analysis.json')

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    for n in names:
        label, colour, width = STYLE[n]
        ax.plot(days, rolling(series[n]['ic'], WINDOW), color=colour, linewidth=width,
                label=f"{label}  (test mean {a['table'][n]['test']['mean_ic']:+.3f})")
    _axes(ax)
    ax.set_title(f'Rolling {WINDOW}-day mean rank IC on the test period', loc='left', color=INK, fontsize=12)
    ax.set_ylabel('rank IC, next-day returns', color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / 'rolling_ic.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=150, gridspec_kw=dict(width_ratios=(1, 1.25)))
    for ax, shown, title in ((axes[0], names, 'All signals'),
                             (axes[1], [n for n in names if n != 'reversal_1d'], 'Zoom: without reversal_1d')):
        for n in shown:
            label, colour, width = STYLE[n]
            ax.plot(days, cumulative(series[n]['net']), color=colour, linewidth=width,
                    label=f"{label}  (sum {a['table'][n]['test']['sum_net']:+.3f})")
        _axes(ax)
        ax.set_title(title, loc='left', color=INK, fontsize=10.5)
    axes[0].set_ylabel('cumulative net return (sum, no compounding)', color=MUTED, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, 0.075))
    fig.suptitle(f'Cumulative net return on the test period, after {fee:g} bps per side on turnover',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.text(0.01, 0.01, note + '.\nDollar-neutral, unit gross, daily rebalance; no borrow, funding or slippage.',
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(out / 'cumulative_net.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
