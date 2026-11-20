from __future__ import annotations
import json, pathlib, sys
import numpy as np
import pandas as pd
from scipy.stats import beta as _beta, rankdata

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

DESIGN_ORDER = ['independent', 'groups25', 'groups10', 'common16', 'common8', 'common2', 'following']
DESIGN_LABEL = {'independent': 'Independent', 'groups25': '25 groups', 'groups10': '10 groups',
                'common16': r'Shared, $\sigma=16$', 'common8': r'Shared, $\sigma=8$',
                'common2': r'Shared, $\sigma=2$', 'following': 'Following'}
DESIGN_PLAIN = {'independent': 'Independent', 'groups25': '25 groups', 'groups10': '10 groups',
                'common16': 'Shared σ=16', 'common8': 'Shared σ=8', 'common2': 'Shared σ=2',
                'following': 'Following'}
COL = dict(spectral='#2a78d6', grouped='#eb6834', pooled='#1baf7a', reference='#52514e',
           naive='#8a8984', additive='#eda100', scalar='#6250d6')
NAMES = dict(naive='Naive', additive='Additive local', grouped='Grouped local',
             pooled='Uncentred spectral', spectral='Centred spectral', reference='Reference',
             scalar='Trained scalar')

NUMBERS = {}


def save_numbers():
    p = OUT / 'numbers.json'
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(NUMBERS)
    p.write_text(json.dumps(old, indent=2, sort_keys=True))


def load(block):
    path = ROOT / 'results' / block / 'prefix_records.csv'
    if not path.exists():
        path = path.with_suffix('.csv.gz')
    return pd.read_csv(path, dtype={'group_means': str}, low_memory=False)


def beta_reports(success, n, ess, target):
    """Vectorised working-Beta reports."""
    s = ess * success / n
    a, b = 1 + s, 1 + ess - s
    p_pos = _beta.sf(0.5, a, b)
    lo = _beta.ppf(0.025, a, b)
    hi = _beta.ppf(0.975, a, b)
    truth = (target >= 0.5).astype(int)
    side = (a >= b).astype(int)
    conf = np.where(side == 1, p_pos, 1 - p_pos)
    return pd.DataFrame(dict(p_pos=p_pos, conf=conf, side=side, correct=(side == truth).astype(int),
                             width=hi - lo, cover=((lo <= target) & (target <= hi)).astype(int),
                             brier=(p_pos - truth) ** 2, high=(conf >= .95).astype(int)))


def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos, neg = label.sum(), len(label) - label.sum()
    if pos == 0 or neg == 0:
        return np.nan
    r = rankdata(score)
    return (r[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def boot_ci(values_fn, groups, n_boot=2000, seed=1):
    """Stratified world bootstrap: groups is a list of index arrays (strata)."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        stats.append(values_fn(idx))
    return np.percentile(np.asarray(stats), [2.5, 97.5], axis=0)


def fmt(x, d=2):
    return f'{x:.{d}f}'


def write_table(name, header, rows, align):
    lines = [r'\begin{tabular}{' + align + '}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for r in rows:
        if r == 'MID':
            lines.append(r'\midrule')
        else:
            lines.append(' & '.join(r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_grouped(name, header, groups, align):
    """Table with group title rows spanning all columns instead of a label column."""
    ncol = len(align)
    lines = [r'\begin{tabular}{' + align + '}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for k, (title, rows) in enumerate(groups):
        if k:
            lines.append(r'\addlinespace')
        lines.append(r'\multicolumn{%d}{l}{\emph{%s}} \\' % (ncol, title))
        lines += [' & '.join(r) + r' \\' for r in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def style():
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#52514e', 'axes.linewidth': .6,
                         'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': .5,
                         'lines.linewidth': 1.6, 'figure.dpi': 150, 'savefig.dpi': 600,
                         'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
