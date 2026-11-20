"""Sensor noise and swarm size (robustness block)."""
from common import *
rb = load('robustness')
f = rb[rb.prefix == 64].copy()
for m in ['grouped', 'pooled', 'spectral']:
    f['lr_' + m] = np.log(f[m] / f.reference)
groups = []
vl = {'noise05': r'Flip 0.05', 'noise10': r'Flip 0.10', 'N20': r'$N=20$', 'N100': r'$N=100$'}
for v in ['noise05', 'noise10', 'N20', 'N100']:
    rows = []
    for d in ['independent', 'common2', 'following']:
        g = f[(f.variant == v) & (f.design == d)]
        r = [DESIGN_LABEL[d]]
        for m in ['grouped', 'pooled', 'spectral']:
            r.append(fmt(np.exp(g['lr_' + m].median())))
            NUMBERS[f'rob_r50_{m}_{v}_{d}'] = float(np.exp(g['lr_' + m].median()))
        q = np.exp(np.quantile(np.abs(g.lr_spectral), .9)); r.append(fmt(q))
        NUMBERS[f'rob_q90_spectral_{v}_{d}'] = float(q)
        rep = {m: beta_reports(g.success.values, g.n.values, g[m].values, g.target.values) for m in ['grouped', 'spectral']}
        for m in ['grouped', 'spectral']:
            r.append(fmt(rep[m].brier.mean(), 3))
            NUMBERS[f'rob_brier_{m}_{v}_{d}'] = float(rep[m].brier.mean())
        rows.append(r)
    groups.append(({'noise05': 'Sensor flip rate 0.05', 'noise10': 'Sensor flip rate 0.10', 'N20': '20 robots, noiseless',
                    'N100': '100 robots, noiseless'}[v], rows))
write_grouped('robustness', ['Deployment', 'Grouped', 'Uncentred', 'Centred', r'Centred $Q_{.9}$',
                             'Brier grouped', 'Brier centred'], groups, 'lrrrrrr')
save_numbers()
print('robustness done')
