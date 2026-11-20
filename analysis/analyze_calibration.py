"""Conformal calibration with full-pipeline world bootstrap.

Each replicate resamples worlds with replacement within deployment x scale
cells, assigns every distinct sampled world to the calibration or the test
fold, refits all quantiles and evaluates the test fold. Reported intervals
therefore include calibration-set variation.

Scopes:   design   - one quantile per deployment design (design known)
          agnostic - one quantile for all seven designs
Kinds:    point    - one quantile per decision time
          joint    - one quantile for the maximum score over the eight times
          bonf     - one quantile per decision time at beta/8 (union bound over times)
Robust:   False    - calibrate on the mixture of field scales
          True     - max over the scale-specific quantiles (Prop. 5)
Methods:  grouped, spectral (centred), combined = max of both, each at beta/2
"""
from common import *
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from swarm_ess import calibration as cal, LOOKS

BETA = 0.05
R = 400
SCEN = ['mixture', 'xi0', 'xi16', 'xi24']


def prepare(bank):
    g = bank.sort_values(['design', 'xi', 'fill', 'world', 'prefix'])
    T = len(LOOKS)
    first = g[g.prefix == LOOKS[0]]
    d = dict(design=first.design.values, xi=first.xi.values, cell=(first.design + '_' + first.xi.astype(str)).values)
    for c in ['reference', 'n', 'success', 'target', 'grouped', 'spectral']:
        d[c] = g[c].values.reshape(-1, T)
    return d


def quantiles(S, kind, beta, strata=None):
    def q_of(s):
        if kind == 'point':
            return np.array([cal.conformal_quantile(s[:, t], beta) for t in range(s.shape[1])])
        if kind == 'bonf':
            return np.array([cal.conformal_quantile(s[:, t], beta / s.shape[1]) for t in range(s.shape[1])])
        return np.full(s.shape[1], cal.conformal_quantile(s.max(1), beta))
    if strata is None:
        return q_of(S)
    return np.max([q_of(S[strata == x]) for x in np.unique(strata)], axis=0)


def replicate(args):
    rep, data = args
    rng = np.random.default_rng(90000 + rep)
    xi, des, cell = data['xi'], data['design'], data['cell']
    cal_idx, test_idx = [], []
    for c in np.unique(cell[xi <= 16]):
        pool = np.flatnonzero(cell == c)
        draw = rng.choice(pool, len(pool), replace=True)
        uniq = np.unique(draw)
        lab = dict(zip(uniq, rng.integers(0, 2, len(uniq))))
        for w in draw:
            (cal_idx if lab[w] == 0 else test_idx).append(w)
    cal_idx, test_idx = np.array(cal_idx), np.array(test_idx)
    out_idx = np.concatenate([rng.choice(np.flatnonzero(cell == c), (cell == c).sum(), replace=True)
                              for c in np.unique(cell[xi == 24])])
    logref = np.log(data['reference'])
    logn = np.log(data['n'])
    rows = []
    for scope in ('design', 'agnostic'):
        units = [(d, des == d) for d in DESIGN_ORDER] if scope == 'design' else [('all', np.ones(len(xi), bool))]
        for kind in ('point', 'joint', 'bonf'):
            for robust in (False, True):
                est_log = {m: np.full(data['grouped'].shape, np.nan) for m in ('grouped', 'spectral', 'combined')}
                for name, mask in units:
                    ci = cal_idx[mask[cal_idx]]
                    strata = xi[ci] if robust else None
                    q = {}
                    for m in ('grouped', 'spectral'):
                        S = np.log(data[m][ci]) - logref[ci]
                        q[m] = quantiles(S, kind, BETA, strata)
                        q[m + '_h'] = quantiles(S, kind, BETA / 2, strata)
                    for m in ('grouped', 'spectral'):
                        est_log[m][mask] = np.log(data[m][mask]) - q[m][None, :]
                    est_log['combined'][mask] = np.maximum(np.log(data['grouped'][mask]) - q['grouped_h'][None, :],
                                                           np.log(data['spectral'][mask]) - q['spectral_h'][None, :])
                for m, L in est_log.items():
                    L = np.clip(L, 0, logn)
                    for sc in SCEN:
                        idx = {'mixture': test_idx, 'xi0': test_idx[xi[test_idx] == 0],
                               'xi16': test_idx[xi[test_idx] == 16], 'xi24': out_idx}[sc]
                        lr = L[idx] - logref[idx]
                        rec = dict(rep=rep, scope=scope, kind=kind, robust=robust, method=m, scenario=sc, design='all',
                                   viol64=float((lr[:, -1] > 1e-12).mean()), viol_any=float((lr > 1e-12).any(1).mean()),
                                   georatio64=float(np.exp(lr[:, -1].mean())), meanratio64=float(np.exp(lr[:, -1]).mean()),
                                   georatio_all=float(np.exp(lr.mean())))
                        if sc in ('mixture', 'xi16'):
                            r = beta_reports(data['success'][idx, -1], data['n'][idx, -1], np.exp(L[idx, -1]),
                                             data['target'][idx, -1])
                            rec.update(brier64=float(r.brier.mean()), width64=float(r.width.mean()),
                                       high64=float(r.high.mean()))
                        rows.append(rec)
                        if scope == 'design' and sc == 'mixture':
                            for dname in DESIGN_ORDER:
                                di = idx[des[idx] == dname]
                                lr = L[di] - logref[di]
                                rows.append(dict(rep=rep, scope=scope, kind=kind, robust=robust, method=m,
                                                 scenario=sc, design=dname,
                                                 viol64=float((lr[:, -1] > 1e-12).mean()),
                                                 viol_any=float((lr > 1e-12).any(1).mean()),
                                                 georatio64=float(np.exp(lr[:, -1].mean()))))
    return rows


if __name__ == '__main__':
    data = prepare(load('bank'))
    rows = []
    with ProcessPoolExecutor(20) as ex:
        for r in ex.map(replicate, [(k, data) for k in range(R)], chunksize=5):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'calibration_bootstrap.csv', index=False)
    df = df[df.design == 'all']
    metrics = ['viol64', 'viol_any', 'georatio64', 'meanratio64', 'georatio_all', 'brier64', 'width64', 'high64']
    keys = ['scope', 'kind', 'robust', 'method', 'scenario']
    summ = df.groupby(keys)[metrics].agg(['mean', lambda s: s.quantile(.025), lambda s: s.quantile(.975)])
    for k, r in summ.iterrows():
        tag = '_'.join([k[0], k[1], 'robust' if k[2] else 'mix', k[3], k[4]])
        for met in metrics:
            v = r[(met, 'mean')]
            if not np.isnan(v):
                NUMBERS[f'cal_{met}_{tag}'] = [float(v), float(r[(met, '<lambda_0>')]), float(r[(met, '<lambda_1>')])]
    piv = df.pivot_table(index=['rep', 'scope', 'kind', 'robust', 'scenario'], columns='method', values=metrics)
    for met in metrics:
        for a, b in [('spectral', 'grouped'), ('combined', 'grouped'), ('combined', 'spectral')]:
            if (met, a) not in piv.columns:
                continue
            d = (np.log(piv[(met, a)] / piv[(met, b)]) if met.startswith('georatio') else piv[(met, a)] - piv[(met, b)]).dropna()
            for k, g in d.groupby(level=['scope', 'kind', 'robust', 'scenario']):
                tag = '_'.join([k[0], k[1], 'robust' if k[2] else 'mix', k[3]])
                NUMBERS[f'calc_{met}_{a}_{b}_{tag}'] = [float(g.mean()), float(g.quantile(.025)), float(g.quantile(.975))]
    save_numbers()

    def c(scope, kind, rob, m, sc, met, d=2, pct=False, ci=False):
        v, lo, hi = NUMBERS[f'cal_{met}_{scope}_{kind}_{rob}_{m}_{sc}']
        k = 100 if pct else 1
        s = f'{v*k:.{d}f}'
        return s + (f' [{lo*k:.{d}f}, {hi*k:.{d}f}]' if ci else '')
    ML = {'grouped': 'Grouped local', 'spectral': 'Centred spectral', 'combined': 'Combined'}
    groups = []
    KIND_TITLE = {'point': 'single look', 'bonf': 'all eight looks, Bonferroni ($\\beta/8$ per look)',
                  'joint': 'all eight looks, maximum score'}
    for scope in ('design', 'agnostic'):
        for kind in ('point', 'bonf', 'joint'):
            title = {'design': 'Per deployment', 'agnostic': 'One quantile for all deployments'}[scope] + ', ' + KIND_TITLE[kind]
            rows = []
            for m in (('grouped', 'spectral') if kind == 'bonf' else ('grouped', 'spectral', 'combined')):
                vm = 'viol64' if kind == 'point' else 'viol_any'
                v, lo, hi = NUMBERS[f'cal_georatio64_{scope}_{kind}_mix_{m}_mixture']
                rows.append([ML[m], c(scope, kind, 'mix', m, 'mixture', vm, 1, True), f'{v:.3f}', f'[{lo:.3f}, {hi:.3f}]',
                             c(scope, kind, 'mix', m, 'mixture', 'georatio_all', 3),
                             c(scope, kind, 'mix', m, 'mixture', 'brier64', 3), c(scope, kind, 'mix', m, 'mixture', 'width64', 3)])
            groups.append((title, rows))
    write_grouped('calibration_main', ['ESS', r'Viol.\ (\%)', 'Ratio 64', '95\\% CI', 'Ratio all', 'Brier', 'Width'],
                  groups, 'lrrcrrr')
    groups = []
    for kind in ('point', 'bonf', 'joint'):
        for rob in ('mix', 'robust'):
            rows = []
            for m in (('grouped', 'spectral') if kind == 'bonf' else ('grouped', 'spectral', 'combined')):
                vm = 'viol64' if kind == 'point' else 'viol_any'
                r = [ML[m]]
                for sc in SCEN:
                    r += [c('design', kind, rob, m, sc, vm, 1, True), c('design', kind, rob, m, sc, 'georatio64', 3)]
                rows.append(r)
            groups.append(({'point': 'Single look', 'bonf': 'All looks, Bonferroni', 'joint': 'All looks, maximum score'}[kind] + (', scale-robust quantile' if rob == 'robust' else ', mixture quantile'), rows))
    lines_header = ['ESS', 'Viol.', 'Ratio', 'Viol.', 'Ratio', 'Viol.', 'Ratio', 'Viol.', 'Ratio']
    write_grouped('calibration_shift', lines_header, groups, 'lrrrrrrrr')
    t = (TAB / 'calibration_shift.tex').read_text(encoding='utf-8')
    t = t.replace(r'\toprule', r'\toprule' + '\n' + r' & \multicolumn{2}{c}{Mixture} & \multicolumn{2}{c}{$\xi=0$} & \multicolumn{2}{c}{$\xi=16$} & \multicolumn{2}{c}{$\xi=24$} \\' + '\n' + r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}', 1)
    (TAB / 'calibration_shift.tex').write_text(t, encoding='utf-8')

    style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    titles = {'mixture': 'Calibration mixture', 'xi0': r'Only $\xi=0$', 'xi16': r'Only $\xi=16$', 'xi24': r'Out of range, $\xi=24$'}
    for ax, sc in zip(axes, SCEN):
        for m, mk in (('grouped', 'o'), ('spectral', 's'), ('combined', 'D')):
            for rob, fill in (('mix', False), ('robust', True)):
                v, lo, hi = NUMBERS[f'cal_viol64_design_point_{rob}_{m}_{sc}']
                g, glo, ghi = NUMBERS[f'cal_georatio64_design_point_{rob}_{m}_{sc}']
                col = COL[m] if m != 'combined' else COL['scalar']
                ax.errorbar(100 * v, g, xerr=[[max(0.0, 100 * (v - lo))], [max(0.0, 100 * (hi - v))]], yerr=[[max(0.0, g - glo)], [max(0.0, ghi - g)]], fmt=mk, ms=4.5,
                            color=col, mfc=col if fill else 'white', elinewidth=.8)
        ax.axvline(5, color=COL['reference'], ls='--', lw=1)
        ax.set_title(titles[sc]); ax.set_xlabel('Violations at step 64 (%)')
    axes[0].set_ylabel('Geometric mean ESS / reference')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL['grouped'], marker='o', ls='', label='Grouped local'),
         Line2D([], [], color=COL['spectral'], marker='s', ls='', label='Centred spectral'),
         Line2D([], [], color=COL['scalar'], marker='D', ls='', label='Combined'),
         Line2D([], [], color='#52514e', marker='o', mfc='white', ls='', label='Mixture quantile'),
         Line2D([], [], color='#52514e', marker='o', ls='', label='Scale-robust quantile')]
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(FIG / 'calibration.png'); fig.savefig(FIG / 'calibration.pdf'); plt.close(fig)
    print('calibration done')
