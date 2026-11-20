"""Ablation: per-robot local ESS (truncated at the first non-positive lag, at most
sqrt(n) lags) versus the swarm-pooled local ESS of Equation (eq:local), on the
torus with sensor flips. Seeds come from the root seed via SeedSequence."""
from common import *
from concurrent.futures import ProcessPoolExecutor
from swarm_ess import simulator, estimators as est
from experiments_seed import seed_for

DES = {'independent': (50, 2.0, False), 'groups10': (10, 2.0, False), 'common2': (1, 2.0, False),
       'following': (1, 2.0, True)}


def run(job):
    w = simulator.simulate(job['seed'], job['xi'], job['fill'], job['groups'], job['sigma'], job['follow'],
                           flip_prob=job['flip'])
    ref, _ = est.reference_ess(w.field, w.x, job['flip'])
    per = np.array([est.local_ess(s) for s in w.y])
    pool = est.pooled_local_ess(w.y)
    out = dict(job, reference=ref)
    for name, loc in (('per', per), ('pool', pool)):
        sp = est.spectral_ess(w.y, w.x, 128, loc, flip_prob=job['flip'])
        out['grouped_' + name] = sp['grouped']; out['spectral_' + name] = sp['ess']
    return out


if __name__ == '__main__':
    jobs = []
    for a, flip in enumerate((0.0, 0.05, 0.10)):
        for b, (d, (g, s, f)) in enumerate(DES.items()):
            for c, xi in enumerate((0.0, 8.0, 16.0)):
                for w in range(50):
                    jobs.append(dict(flip=flip, design=d, groups=g, sigma=s, follow=f, xi=xi,
                                     fill=(0.30, 0.48)[w % 2], world=w, seed=seed_for(6, a, b, c, w)))
    with ProcessPoolExecutor(22) as ex:
        d = pd.DataFrame(list(ex.map(run, jobs, chunksize=4)))
    d.to_csv(OUT / 'ablation_local.csv', index=False)
    for m in ('grouped_per', 'grouped_pool', 'spectral_per', 'spectral_pool'):
        d['lr_' + m] = np.log(d[m] / d.reference)
    rows = []
    for flip in (0.0, 0.05, 0.10):
        for dname in DES:
            g = d[(d.flip == flip) & (d.design == dname)]
            r = {m: float(np.exp(g['lr_' + m].median())) for m in ('grouped_per', 'grouped_pool', 'spectral_per', 'spectral_pool')}
            for m, v in r.items():
                NUMBERS[f'abl_{m}_f{int(flip * 100)}_{dname}'] = v
            rows.append([f'{flip:.2f}', DESIGN_LABEL[dname]] + [fmt(r[m]) for m in ('grouped_per', 'grouped_pool', 'spectral_per', 'spectral_pool')])
        g = d[d.flip == flip]
        for m in ('grouped_per', 'grouped_pool', 'spectral_per', 'spectral_pool'):
            NUMBERS[f'abl_{m}_f{int(flip * 100)}_q90'] = float(np.exp(np.quantile(np.abs(g['lr_' + m]), .9)))
    write_table('ablation_local', ['Flip rate', 'Deployment', 'Grouped, per robot', 'Grouped, pooled',
                                   'Centred, per robot', 'Centred, pooled'], rows, 'llrrrr')
    save_numbers()
    print(pd.DataFrame(rows).to_string())
