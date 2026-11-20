"""Fixed-budget accuracy, prefix behaviour and confidence quality (bank block)."""
from common import *
import matplotlib.pyplot as plt

style()
bank = load('bank')
inr = bank[bank.xi <= 16].copy()
METHODS = ['naive', 'additive', 'grouped', 'pooled', 'spectral']
for m in METHODS:
    bank['lr_' + m] = np.log(bank[m] / bank.reference)
inr = bank[bank.xi <= 16]
fin = inr[inr.prefix == 64]

# ---------------------------------------------------------------- Prop. 1
ind = bank[(bank.design == 'independent') & (bank.prefix == 64)]
for xi, g in ind.groupby('xi'):
    NUMBERS[f'cross_ind_xi{int(xi)}_mean'] = float(g.ref_cross_raw.mean())
    NUMBERS[f'cross_ind_xi{int(xi)}_se'] = float(g.ref_cross_raw.std(ddof=1) / np.sqrt(len(g)))
for d in ['common2', 'following']:
    g = bank[(bank.design == d) & (bank.prefix == 64) & (bank.xi == 16)]
    NUMBERS[f'cross_{d}_xi16_mean'] = float(g.ref_cross_raw.mean())
    NUMBERS[f'cross_{d}_xi16_se'] = float(g.ref_cross_raw.std(ddof=1) / np.sqrt(len(g)))
g = bank[(bank.prefix == 64) & (bank.xi == 16)]
NUMBERS['mean_D_xi16_independent'] = float(g[g.design == 'independent'].ref_D.mean())

# ---------------------------------------------------------- Table 1
rows = []
for d in DESIGN_ORDER:
    for xi in (0, 8, 16):
        g = fin[(fin.design == d) & (fin.xi == xi)]
        r = {m: np.exp(g['lr_' + m].median()) for m in ['grouped', 'pooled', 'spectral']}
        q90 = {m: np.exp(np.quantile(np.abs(g['lr_' + m]), .9)) for m in ['grouped', 'spectral']}
        rows.append([DESIGN_LABEL[d] if xi == 0 else '', str(xi), fmt(r['grouped']), fmt(r['pooled']),
                     fmt(r['spectral']), fmt(q90['grouped']), fmt(q90['spectral'])])
        for m in ['grouped', 'pooled', 'spectral']:
            NUMBERS[f'r50_{m}_{d}_xi{xi}'] = float(r[m])
        NUMBERS[f'q90_spectral_{d}_xi{xi}'] = float(q90['spectral'])
        NUMBERS[f'q90_grouped_{d}_xi{xi}'] = float(q90['grouped'])
    if d != DESIGN_ORDER[-1]:
        rows.append('MID')
write_table('ess_accuracy', ['Deployment', r'$\xi$', 'Grouped', 'Uncentred', 'Centred',
                             r'Grouped $Q_{.9}$', r'Centred $Q_{.9}$'], rows, 'llrrrrr')
for m in METHODS:
    NUMBERS[f'abslog_mean_{m}_final'] = float(np.abs(fin['lr_' + m]).mean())
    NUMBERS[f'abslog_median_{m}_final'] = float(np.abs(fin['lr_' + m]).median())
    NUMBERS[f'frac_within2_{m}_final'] = float((np.abs(fin['lr_' + m]) <= np.log(2)).mean())

# ------------------------------------------------------------ Figure: deployment
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
xs = np.arange(len(DESIGN_ORDER))
for ax, xi in zip(axes, (0, 8, 16)):
    for k, m in enumerate(['grouped', 'pooled', 'spectral']):
        med, lo, hi = [], [], []
        for d in DESIGN_ORDER:
            v = np.exp(fin[(fin.design == d) & (fin.xi == xi)]['lr_' + m])
            med.append(v.median()); lo.append(v.quantile(.25)); hi.append(v.quantile(.75))
        off = (k - 1) * 0.18
        ax.errorbar(xs + off, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt='o',
                    ms=4, color=COL[m], elinewidth=1, capsize=0, label=NAMES[m])
    ax.axhline(1, color=COL['reference'], ls='--', lw=1)
    ax.set_yscale('log'); ax.set_title(rf'Field scale $\xi={xi}$')
    ax.set_xticks(xs); ax.set_xticklabels([DESIGN_PLAIN[d] for d in DESIGN_ORDER], rotation=50, ha='right')
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel('ESS / reference (median, IQR)')
axes[0].legend(frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(FIG / 'deployment_accuracy.png'); fig.savefig(FIG / 'deployment_accuracy.pdf'); plt.close(fig)

# ------------------------------------------------------------ Figure: prefixes
shared = inr[inr.design.isin(['common2', 'common8', 'common16', 'following'])]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
for m in ['grouped', 'pooled', 'spectral']:
    agg = shared.groupby('prefix')['lr_' + m]
    axes[0].plot(agg.median().index, np.exp(agg.median()), '-o', ms=3.5, color=COL[m], label=NAMES[m])
    axes[1].plot(agg.quantile(.95).index, np.exp(agg.quantile(.95)), '-o', ms=3.5, color=COL[m], label=NAMES[m])
for ax, t in zip(axes, ['Median', '95th percentile']):
    ax.axhline(1, color=COL['reference'], ls='--', lw=1)
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64']); ax.set_xlabel('Observations per robot')
    ax.set_title(f'{t} of ESS / reference')
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / 'prefix_accuracy.png'); fig.savefig(FIG / 'prefix_accuracy.pdf'); plt.close(fig)
for m in ['grouped', 'pooled', 'spectral']:
    agg = shared.groupby('prefix')['lr_' + m]
    for t in (1, 8, 16, 64):
        NUMBERS[f'prefix_q95_{m}_t{t}'] = float(np.exp(agg.quantile(.95)[t]))
        NUMBERS[f'prefix_med_{m}_t{t}'] = float(np.exp(agg.median()[t]))
    NUMBERS[f'prefix_q95_{m}_max'] = float(np.exp(agg.quantile(.95).max()))
st = shared.groupby('prefix').status.apply(lambda s: (s == 'constant_patch').mean())
NUMBERS['constant_patch_t1'] = float(st[1]); NUMBERS['constant_patch_t64'] = float(st[64])

# ------------------------------------------------------- confidence quality
fin = fin.reset_index(drop=True)
# trained scalar: c * grouped, Brier-optimal on the other half of worlds (cross-fitting)
grid = np.concatenate([[0], 10 ** np.linspace(-3, 0, 121)])
def brier_for(c, sub):
    e = np.maximum(c * sub.grouped.values, 1e-9)
    return beta_reports(sub.success.values, sub.n.values, e, sub.target.values).brier.mean()
fin['scalar'] = np.nan
cs = {}
for parity in (0, 1):
    train = fin[fin.world % 2 != parity]
    best = min(grid, key=lambda c: brier_for(c, train))
    cs[parity] = best
    mask = fin.world % 2 == parity
    fin.loc[mask, 'scalar'] = np.maximum(best * fin.loc[mask, 'grouped'], 1e-9)
NUMBERS['scalar_c_even'] = float(cs[0]); NUMBERS['scalar_c_odd'] = float(cs[1])
CONF = ['naive', 'grouped', 'scalar', 'pooled', 'spectral', 'reference']
rep = {m: beta_reports(fin.success.values, fin.n.values, fin[m].values, fin.target.values) for m in CONF}
strata = [g.index.values for _, g in fin.groupby(['design', 'xi', 'fill'])]
def summary(idx, m):
    r = rep[m].iloc[idx]
    return np.array([r.brier.mean(), r.cover.mean(), r.width.mean(), r.high.mean(),
                     ((r.high == 1) & (r.correct == 0)).mean(), auroc(r.conf, r.correct)])
rows = []
for m in CONF:
    s = summary(np.arange(len(fin)), m)
    for k, nm in enumerate(['brier', 'cover', 'width', 'high', 'falsehigh', 'auroc']):
        NUMBERS[f'conf_{nm}_{m}'] = float(s[k])
    rows.append([NAMES[m], fmt(s[0], 3), fmt(s[1], 3), fmt(s[2], 3), fmt(s[3], 3), fmt(s[4], 3), fmt(s[5], 3)])
write_table('confidence', ['ESS', 'Brier', 'Cover', 'Width', 'High', 'False high', 'AUROC'], rows, 'lrrrrrr')
for other in ['grouped', 'pooled', 'scalar']:
    diff = lambda idx, o=other: summary(idx, 'spectral')[[0, 5]] - summary(idx, o)[[0, 5]]
    ci = boot_ci(diff, strata, n_boot=1000, seed=11)
    pt = diff(np.arange(len(fin)))
    NUMBERS[f'dbrier_spectral_{other}'] = float(pt[0]); NUMBERS[f'dbrier_spectral_{other}_ci'] = list(map(float, ci[:, 0]))
    NUMBERS[f'dauroc_spectral_{other}'] = float(pt[1]); NUMBERS[f'dauroc_spectral_{other}_ci'] = list(map(float, ci[:, 1]))
# per-design brier/falsehigh
rows = []
for d in DESIGN_ORDER:
    idx = fin.index[fin.design == d].values
    vals = []
    for m in ['grouped', 'scalar', 'spectral', 'reference']:
        s = summary(idx, m)
        vals += [s[0], s[4]]
        NUMBERS[f'conf_brier_{m}_{d}'] = float(s[0]); NUMBERS[f'conf_falsehigh_{m}_{d}'] = float(s[4])
        NUMBERS[f'conf_cover_{m}_{d}'] = float(s[1])
    rows.append([DESIGN_LABEL[d]] + [fmt(v, 3) for v in vals])
write_table('confidence_design', ['Deployment', 'Grouped', '', 'Scalar', '', 'Centred', '', 'Reference', ''],
            rows, 'lrrrrrrrr')

# figure: Brier and false-high by deployment
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
xs = np.arange(len(DESIGN_ORDER))
for k, m in enumerate(['grouped', 'scalar', 'spectral', 'reference']):
    b = [NUMBERS[f'conf_brier_{m}_{d}'] for d in DESIGN_ORDER]
    fh = [NUMBERS[f'conf_falsehigh_{m}_{d}'] for d in DESIGN_ORDER]
    ls = '--' if m == 'reference' else '-'
    axes[0].plot(xs, b, ls, marker='o', ms=3.5, color=COL[m], label=NAMES[m])
    axes[1].plot(xs, fh, ls, marker='o', ms=3.5, color=COL[m], label=NAMES[m])
axes[0].set_ylabel('Brier score'); axes[1].set_ylabel('False high-confidence rate')
for ax in axes:
    ax.set_xticks(xs); ax.set_xticklabels([DESIGN_PLAIN[d] for d in DESIGN_ORDER], rotation=40, ha='right')
    ax.grid(axis='x', visible=False)
axes[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / 'confidence_design.png'); fig.savefig(FIG / 'confidence_design.pdf'); plt.close(fig)

# ------------------------------------------- posterior stopping vs certificate
from swarm_ess import stopping as stp
thr = stp.threshold(.05, 8)
recs = []
for (d, xi, fill, w), g in inr.groupby(['design', 'xi', 'fill', 'world']):
    g = g.sort_values('prefix')
    truth = int(g.target.iloc[0] >= .5)
    for m in ['grouped', 'spectral', 'reference']:
        stop = None
        for _, r in g.iterrows():
            rr = beta_reports(np.array([r.success]), np.array([r.n]), np.array([r[m]]), np.array([r.target]))
            if rr.conf[0] >= .95:
                stop = (int(r.prefix), int(rr.side[0] == truth)); break
        recs.append(dict(design=d, xi=xi, fill=fill, world=w, rule='posterior_' + m,
                         stopped=stop is not None, correct=bool(stop and stop[1]), time=stop[0] if stop else 64))
    if g.groups.iloc[0] > 1:
        stop = None
        for _, r in g.iterrows():
            z = np.array([float(v) for v in r.group_means.split(';')])
            side = stp.certified_side(stp.mixture_log_e(z), thr)
            if side is not None:
                stop = (int(r.prefix), int(side == truth)); break
        recs.append(dict(design=d, xi=xi, fill=fill, world=w, rule='certificate',
                         stopped=stop is not None, correct=bool(stop and stop[1]), time=stop[0] if stop else 64))
ps = pd.DataFrame(recs)
ps['false'] = ps.stopped & ~ps.correct
ps['near'] = ps.fill.isin([.48, .52])
ps.to_csv(OUT / 'bank_stopping.csv', index=False)
groups = []
RL = {'posterior_grouped': 'Posterior, grouped local', 'posterior_spectral': 'Posterior, centred',
      'posterior_reference': 'Posterior, reference', 'certificate': 'Group certificate'}
for d in ['independent', 'groups25', 'groups10', 'common2', 'following']:
    rows = []
    for rule in ['posterior_grouped', 'posterior_spectral', 'posterior_reference', 'certificate']:
        g = ps[(ps.design == d) & (ps.rule == rule)]
        if not len(g):
            continue
        e, nt = g[~g.near], g[g.near]
        vals = [e.correct.mean(), e['false'].mean(), nt.correct.mean(), nt['false'].mean(), g.time.mean()]
        for k, nm in enumerate(['easy_correct', 'easy_false', 'near_correct', 'near_false', 'time']):
            NUMBERS[f'stop_{nm}_{rule}_{d}'] = float(vals[k])
        rows.append([RL[rule]] + [fmt(v, 2) for v in vals[:4]] + [fmt(vals[4], 1)])
    groups.append((DESIGN_LABEL[d], rows))
write_grouped('posterior_stopping', ['Rule', 'Easy correct', 'Easy wrong', 'Near-tie correct', 'Near-tie wrong', 'Mean time'],
              groups, 'lrrrrr')
for rule in ['posterior_grouped', 'posterior_spectral', 'posterior_reference', 'certificate']:
    g = ps[ps.rule == rule]
    NUMBERS[f'stop_all_false_{rule}'] = float(g['false'].mean())
    NUMBERS[f'stop_all_n_{rule}'] = int(len(g))
save_numbers()
print('bank analysis done')
