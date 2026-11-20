"""Bounded-arena validation: accuracy, confidence, calibration transfer, robustness."""
from common import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from swarm_ess import LOOKS, arena, calibration as cal

style()
A = load('arena')
R = load('arena_robust') if (ROOT / 'results' / 'arena_robust' / 'prefix_records.csv').exists() else None
B = load('bank')
FLOOR_LABEL = {'random': 'Random tiles', 'cluster2': r'Clustered, $\ell=2$', 'cluster4': r'Clustered, $\ell=4$'}
FLOOR_PLAIN = {'random': 'Random', 'cluster2': 'Clustered ℓ=2', 'cluster4': 'Clustered ℓ=4'}
LAUNCH_LABEL = {'uniform': 'Scattered', 'nest': 'Nest'}
for d in [d for d in (A, R) if d is not None]:
    for m in ('naive', 'grouped', 'spectral'):
        d['lr_' + m] = np.log(d[m] / d.reference)

# ---------------------------------------------------------------- accuracy
fin = A[A.prefix == 64]
rows = []
for pool in ('swarm', 'local'):
    grp = []
    for kind in ('random', 'cluster2', 'cluster4'):
        for launch in ('uniform', 'nest'):
            g = fin[(fin.pool == pool) & (fin.floor == kind) & (fin.launch == launch)]
            r = {m: float(np.exp(g['lr_' + m].median())) for m in ('grouped', 'spectral')}
            q = {m: float(np.exp(np.quantile(np.abs(g['lr_' + m]), .9))) for m in ('grouped', 'spectral')}
            a = A[(A.pool == pool) & (A.floor == kind) & (A.launch == launch)]
            q95 = a.groupby('prefix').lr_spectral.quantile(.95)
            grp.append([FLOOR_LABEL[kind], LAUNCH_LABEL[launch], f'{g.reference.median():.0f}',
                        fmt(r['grouped']), fmt(r['spectral']), fmt(q['grouped']), fmt(q['spectral']),
                        fmt(float(np.exp(q95.max())))])
            for m in ('grouped', 'spectral'):
                NUMBERS[f'arena_{pool}_{kind}_{launch}_r50_{m}'] = r[m]
                NUMBERS[f'arena_{pool}_{kind}_{launch}_q90_{m}'] = q[m]
            NUMBERS[f'arena_{pool}_{kind}_{launch}_q95max_spectral'] = float(np.exp(q95.max()))
            NUMBERS[f'arena_{pool}_{kind}_{launch}_ref'] = float(g.reference.median())
            NUMBERS[f'arena_{pool}_{kind}_{launch}_n'] = float(g.n.median())
    rows.append(('Whole swarm (all readings)' if pool == 'swarm' else 'One robot (own and received readings)', grp))
write_grouped('arena_accuracy', ['Floor', 'Launch', r'$e_{\rm ref}$', 'Grouped', 'Centred',
                                 r'Grouped $Q_{.9}$', r'Centred $Q_{.9}$', r'Centred $Q_{.95}^{\max}$'],
              rows, 'llrrrrrr')
for pool in ('swarm', 'local'):
    g = fin[fin.pool == pool]
    for m in ('naive', 'grouped', 'spectral'):
        NUMBERS[f'arena_{pool}_within2_{m}'] = float((np.abs(g['lr_' + m]) <= np.log(2)).mean())
        NUMBERS[f'arena_{pool}_abslog_{m}'] = float(np.abs(g['lr_' + m]).median())
for pool in ('swarm', 'local'):
    for kind in ('random', 'cluster2', 'cluster4'):
        a = A[(A.pool == pool) & (A.floor == kind) & (A.launch == 'nest')]
        for m in ('grouped', 'spectral'):
            q = a.groupby('prefix')['lr_' + m].quantile(.95)
            NUMBERS[f'arena_{pool}_{kind}_nest_q95t1_{m}'] = float(np.exp(q[1]))
            NUMBERS[f'arena_{pool}_{kind}_nest_q95max_{m}'] = float(np.exp(q.max()))
        NUMBERS[f'arena_{pool}_{kind}_nest_const_t1'] = float((a[a.prefix == 1].status == 'constant_patch').mean())
loc = fin[fin.pool == 'local']
NUMBERS['arena_local_sources_median'] = float(loc.sources.median())
NUMBERS['arena_local_n_median'] = float(loc.n.median())
NUMBERS['arena_local_n_share'] = float((loc.n / (20 * 64)).median())
NUMBERS['arena_contacts_median'] = float(fin[fin.pool == 'swarm'].contacts.median())
for launch in ('uniform', 'nest'):
    g = loc[loc.launch == launch]
    NUMBERS[f'arena_local_{launch}_sources_median'] = float(g.sources.median())
    NUMBERS[f'arena_local_{launch}_n_median'] = float(g.n.median())

# ---------------------------------------------------------------- confidence
rows = []
CONF = ['naive', 'grouped', 'spectral', 'reference']
for pool in ('swarm', 'local'):
    g = fin[fin.pool == pool].reset_index(drop=True)
    grp = []
    for m in CONF:
        r = beta_reports(g.success.values, g.n.values, g[m].values, g.target.values)
        vals = [r.brier.mean(), r.cover.mean(), r.width.mean(), ((r.high == 1) & (r.correct == 0)).mean(),
                auroc(r.conf, r.correct)]
        for k, nm in enumerate(['brier', 'cover', 'width', 'falsehigh', 'auroc']):
            NUMBERS[f'arena_conf_{pool}_{nm}_{m}'] = float(vals[k])
        grp.append([NAMES[m], fmt(vals[0], 3), fmt(vals[1], 3), fmt(vals[2], 3), fmt(vals[3], 3), fmt(vals[4], 3)])
    # nest launch only, false high
    for m in CONF:
        h = g[g.launch == 'nest'].reset_index(drop=True)
        r = beta_reports(h.success.values, h.n.values, h[m].values, h.target.values)
        NUMBERS[f'arena_conf_{pool}_nest_falsehigh_{m}'] = float(((r.high == 1) & (r.correct == 0)).mean())
        NUMBERS[f'arena_conf_{pool}_nest_cover_{m}'] = float(r.cover.mean())
    rows.append(('Whole swarm' if pool == 'swarm' else 'One robot', grp))
write_grouped('arena_confidence', ['ESS', 'Brier', 'Cover', 'Width', 'False high', 'AUROC'], rows, 'lrrrrr')

# ------------------------------------------- calibration transferred from the torus bank
bank = B[B.xi <= 16].copy()
def robust_quantiles(df, m, beta=0.05):
    q = {}
    for t in LOOKS:
        s = df[df.prefix == t]
        q[t] = max(cal.conformal_quantile(np.log(g[m] / g.reference), beta) for _, g in s.groupby('xi'))
    return q
QB = {m: robust_quantiles(bank, m) for m in ('grouped', 'spectral')}
# quantiles refitted on half of the arena worlds (scattered and nest pooled, all floors)
def arena_quantiles(df, m, beta=0.05):
    q = {}
    for t in LOOKS:
        s = df[df.prefix == t]
        q[t] = max(cal.conformal_quantile(np.log(g[m] / g.reference), beta) for _, g in s.groupby('floor'))
    return q
def evaluate(df, q, m):
    out = {}
    viol, ratio = [], []
    for t in LOOKS:
        s = df[df.prefix == t]
        e = cal.apply(s[m].values, q[t], s.n.values)
        v = e > s.reference.values * (1 + 1e-9)
        viol.append(v.mean()); ratio.append(np.exp(np.mean(np.log(e / s.reference.values))))
    s = df[df.prefix == 64]
    e = cal.apply(s[m].values, q[64], s.n.values)
    rep = beta_reports(s.success.values, s.n.values, e, s.target.values)
    return dict(viol64=viol[-1], violmax=max(viol), ratio64=ratio[-1], cover=rep.cover.mean(),
                falsehigh=((rep.high == 1) & (rep.correct == 0)).mean(), brier=rep.brier.mean())
rows = []
for pool in ('swarm', 'local'):
    df = A[A.pool == pool]
    grp = []
    for m in ('grouped', 'spectral'):
        r = evaluate(df, QB[m], m)
        for k, v in r.items():
            NUMBERS[f'arena_transfer_{pool}_{m}_{k}'] = float(v)
        # split by world parity: fit on even worlds of the arena, test on odd
        tr, te = df[df.world % 2 == 0], df[df.world % 2 == 1]
        r2 = evaluate(te, arena_quantiles(tr, m), m)
        for k, v in r2.items():
            NUMBERS[f'arena_refit_{pool}_{m}_{k}'] = float(v)
        grp.append([NAMES[m], 'torus bank', fmt(100 * r['viol64'], 1), fmt(100 * r['violmax'], 1), fmt(r['ratio64'], 3),
                    fmt(r['cover'], 3), fmt(r['brier'], 3)])
        grp.append([NAMES[m], 'arena (half)', fmt(100 * r2['viol64'], 1), fmt(100 * r2['violmax'], 1), fmt(r2['ratio64'], 3),
                    fmt(r2['cover'], 3), fmt(r2['brier'], 3)])
    rows.append(('Whole swarm' if pool == 'swarm' else 'One robot', grp))
write_grouped('arena_calibration', ['ESS', 'Quantiles from', r'Viol.\ 64', r'Viol.\ max', 'Ratio 64', 'Cover', 'Brier'],
              rows, 'llrrrrr')
for m in ('grouped', 'spectral'):
    NUMBERS[f'bank_robust_q64_{m}'] = float(QB[m][64])

# ---------------------------------------------------------------- robustness
VAR_LABEL = {'base': 'Exact positions', 'flip05': 'Flip rate 0.05', 'flip10': 'Flip rate 0.10',
             'radio5': r'Radio, $\sigma=5$\,cm', 'radio10': r'Radio, $\sigma=10$\,cm', 'odometry': 'Odometry',
             'N50': r'$N=50$', 'slow': r'Speed 2\,cm/s', 'tile5': r'5\,cm tiles'}
rf = R[R.prefix == 64] if R is not None else A.iloc[0:0]
base = fin[fin.fill.isin([0.30, 0.48])]
rows = []
for pool in ('swarm', 'local'):
    grp = []
    for v in ['base'] + list(dict.fromkeys(rf.variant)):
        g = (base if v == 'base' else rf[rf.variant == v])
        g = g[g.pool == pool].reset_index(drop=True)
        cells = []
        for launch in ('uniform', 'nest'):
            h = g[g.launch == launch]
            cells += [fmt(float(np.exp(h.lr_grouped.median()))), fmt(float(np.exp(h.lr_spectral.median())))]
            for m in ('grouped', 'spectral'):
                NUMBERS[f'arobust_{pool}_{v}_{launch}_r50_{m}'] = float(np.exp(h['lr_' + m].median()))
            NUMBERS[f'arobust_{pool}_{v}_{launch}_q90_spectral'] = float(np.exp(np.quantile(np.abs(h.lr_spectral), .9)))
        rg = beta_reports(g.success.values, g.n.values, g.grouped.values, g.target.values)
        rs = beta_reports(g.success.values, g.n.values, g.spectral.values, g.target.values)
        cells += [fmt(rg.cover.mean(), 2), fmt(rs.cover.mean(), 2)]
        NUMBERS[f'arobust_{pool}_{v}_cover_grouped'] = float(rg.cover.mean())
        NUMBERS[f'arobust_{pool}_{v}_cover_spectral'] = float(rs.cover.mean())
        if 'pos_err_median' in g and g.pos_err_median.notna().any():
            NUMBERS[f'arobust_{v}_poserr'] = float(g.pos_err_median.median())
        grp.append([VAR_LABEL[v]] + cells)
    rows.append(('Whole swarm' if pool == 'swarm' else 'One robot', grp))
write_grouped('arena_robust', [r'& \multicolumn{2}{c}{Ratio, scattered} & \multicolumn{2}{c}{Ratio, nest} & '
                              r'\multicolumn{2}{c}{Coverage} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7} Variant',
                              'Grouped', 'Centred', 'Grouped', 'Centred', 'Grouped', 'Centred'],
              rows, 'lrrrrrr')

# ---------------------------------------------------------------- figure
cfg = arena.ArenaConfig(launch='nest')
from experiments_seed import seed_for  # noqa: E402
run = arena.simulate_arena(seed_for(4, 1, 0, 1, 0), 'cluster2', 0.30, cfg)
fig = plt.figure(figsize=(7.0, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.42)
ax = fig.add_subplot(gs[0])
ax.imshow(run.floor.T, origin='lower', extent=[0, 2, 0, 2], cmap=ListedColormap(['#ffffff', '#c9c8c2']),
          interpolation='nearest')
T_SHOW = 20
for i in range(run.x_true.shape[0]):
    ax.plot(run.x_true[i, :T_SHOW, 0], run.x_true[i, :T_SHOW, 1], lw=.6, color=COL['spectral'], alpha=.6)
p = run.x_true[:, T_SHOW - 1]
for i in range(len(p)):
    for j in range(i + 1, len(p)):
        if np.hypot(*(p[i] - p[j])) <= cfg.comm_range:
            ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], lw=.5, ls=':', color=COL['grouped'])
ax.scatter(p[:, 0], p[:, 1], s=9, color=COL['spectral'], zorder=3)
ax.add_patch(plt.Rectangle((0, 0), cfg.nest, cfg.nest, fill=False, ls='--', lw=.8, color=COL['reference']))
ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect('equal'); ax.grid(False)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2]); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'(a) Nest launch, first {T_SHOW} s')
for k, pool in enumerate(('swarm', 'local')):
    ax = fig.add_subplot(gs[k + 1])
    xs = np.array([0, 1, 2.6, 3.6, 5.2, 6.2])
    for j, m in enumerate(('grouped', 'spectral')):
        med, lo, hi = [], [], []
        for kind in ('random', 'cluster2', 'cluster4'):
            for launch in ('uniform', 'nest'):
                v = np.exp(fin[(fin.pool == pool) & (fin.floor == kind) & (fin.launch == launch)]['lr_' + m])
                med.append(v.median()); lo.append(v.quantile(.1)); hi.append(v.quantile(.9))
        ax.errorbar(xs + (j - .5) * .28, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt='o', ms=3.5,
                    color=COL[m], elinewidth=1, capsize=0, label=NAMES[m])
    ax.axhline(1, color=COL['reference'], ls='--', lw=1)
    ax.set_yscale('log'); ax.set_ylim(0.25, 40)
    ax.set_xticks(xs); ax.set_xticklabels(['S', 'N'] * 3, fontsize=7)
    for c, lab in zip((0.5, 3.1, 5.7), ('Random', r'$\ell=2$', r'$\ell=4$')):
        ax.text(c, -0.13, lab, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7)
    ax.grid(axis='x', visible=False)
    ax.set_title('(b) Whole swarm' if pool == 'swarm' else '(c) One robot, local radio')
    if k == 0:
        ax.set_ylabel('ESS / reference')
        ax.legend(frameon=False, loc='upper left', fontsize=6.5)
fig.savefig(FIG / 'arena.png', bbox_inches='tight'); fig.savefig(FIG / 'arena.pdf', bbox_inches='tight'); plt.close(fig)
save_numbers()
print({k: round(v, 3) for k, v in NUMBERS.items() if k.startswith(('arena_conf', 'arena_transfer', 'arena_refit'))})
