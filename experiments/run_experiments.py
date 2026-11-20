"""Run the simulation blocks of the revised study.

  python experiments/run_experiments.py bank       --out results/bank
  python experiments/run_experiments.py stopping   --out results/stopping
  python experiments/run_experiments.py robustness --out results/robustness

Every world seed is derived with numpy SeedSequence from a fixed root seed
and the world's indices, so each block is reproducible and independent.
"""
from __future__ import annotations
import argparse, json, sys, time, pathlib, platform
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarm_ess import LOOKS, simulator, estimators as est

ROOT_SEED = 20261119
DESIGNS = {  # name: (groups, sigma, follow)
    'independent': (50, 2.0, False), 'groups25': (25, 2.0, False), 'groups10': (10, 2.0, False),
    'common2': (1, 2.0, False), 'common8': (1, 8.0, False), 'common16': (1, 16.0, False),
    'following': (1, 2.0, True)}
FILLS = (0.30, 0.48, 0.52, 0.70)


def seed_for(*idx):
    return int(np.random.SeedSequence([ROOT_SEED, *idx]).generate_state(1, dtype=np.uint64)[0])


def jobs_bank(worlds=150):
    out = []
    for di, (d, (g, s, fol)) in enumerate(DESIGNS.items()):
        for xi_i, xi in enumerate((0.0, 8.0, 16.0, 24.0)):
            for fi, fill in enumerate(FILLS):
                for w in range(worlds):
                    out.append(dict(block='bank', design=d, groups=g, sigma=s, follow=fol, n_robots=50,
                                    xi=xi, fill=fill, flip=0.0, world=w,
                                    seed=seed_for(1, di, xi_i, fi, w), spectral=True))
    return out


def jobs_stopping(worlds=100):
    out = []
    configs = [(50, g) for g in (1, 2, 5, 10, 25, 50)] + [(100, g) for g in (10, 25, 50, 100)]
    for ci, (nr, g) in enumerate(configs):
        for xi_i, xi in enumerate((8.0, 16.0)):
            for fi, fill in enumerate((0.30, 0.40, 0.45, 0.48)):
                for w in range(worlds):
                    out.append(dict(block='stopping', design=f'N{nr}_G{g}', groups=g, sigma=2.0, follow=False,
                                    n_robots=nr, xi=xi, fill=fill, flip=0.0, world=w,
                                    seed=seed_for(2, ci, xi_i, fi, w), spectral=False))
    return out


def jobs_robustness(worlds=50):
    out = []
    base = {'independent': None, 'common2': (1, 2.0, False), 'following': (1, 2.0, True)}
    variants = [('noise05', 50, .05), ('noise10', 50, .10), ('N20', 20, 0.0), ('N100', 100, 0.0)]
    for vi, (vname, nr, flip) in enumerate(variants):
        for di, d in enumerate(base):
            g, s, fol = (nr, 2.0, False) if d == 'independent' else base[d]
            for xi_i, xi in enumerate((0.0, 8.0, 16.0)):
                for fi, fill in enumerate((0.30, 0.48)):
                    for w in range(worlds):
                        out.append(dict(block='robustness', variant=vname, design=d, groups=g, sigma=s,
                                        follow=fol, n_robots=nr, xi=xi, fill=fill, flip=flip, world=w,
                                        seed=seed_for(3, vi, di, xi_i, fi, w), spectral=True))
    return out


def run_world(job):
    w = simulator.simulate(job['seed'], job['xi'], job['fill'], job['groups'], job['sigma'],
                           job['follow'], n_robots=job['n_robots'], flip_prob=job['flip'])
    corr = est.field_correlation(w.field)
    f_z = float(w.field.mean())
    target = est.target_rate(w.field, job['flip'])
    rows = []
    for t in LOOKS:
        yp, xp = w.y[:, :t], w.x[:, :t]
        n = yp.size
        row = dict(job, prefix=t, n=n, success=int(yp.sum()), f_z=f_z, target=target)
        gm = [yp[w.group == g].mean() for g in range(job['groups'])]
        row['group_means'] = ';'.join(f'{v:.6f}' for v in gm)
        if job['spectral']:
            loc = est.pooled_local_ess(yp)
            new = est.spectral_ess(yp, xp, 128, loc, centered=True, flip_prob=job['flip'])
            old = est.spectral_ess(yp, xp, 128, loc, centered=False, flip_prob=job['flip'])
            ref, D = est.reference_ess(w.field, xp, job['flip'], corr)
            row.update(naive=float(n), additive=float(min(n, loc.sum())), grouped=new['grouped'],
                       spectral=new['ess'], pooled=old['ess'], reference=ref, ref_D=D,
                       amp=new['amplitude'], length=new['length'], status=new['status'],
                       amp_pooled=old['amplitude'], cross=new['cross'], cross_pooled=old['cross'])
            if t == LOOKS[-1]:
                # raw cross-robot reference covariance (unclipped) for Proposition 1
                total = D
                within = 0.0
                for i in range(w.y.shape[0]):
                    _, Di = est.reference_ess(w.field, xp[i:i + 1], 0.0, corr)
                    within += Di
                row['ref_cross_raw'] = total - within if job['flip'] == 0 else np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('block', choices=['bank', 'stopping', 'robustness'])
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--worlds', type=int, default=None)
    a = ap.parse_args()
    jobs = {'bank': jobs_bank, 'stopping': jobs_stopping, 'robustness': jobs_robustness}[a.block]
    jobs = jobs() if a.worlds is None else jobs(a.worlds)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(run_world, jobs, chunksize=4), 1):
            rows += r
            if i % 500 == 0:
                print(f'{a.block}: {i}/{len(jobs)} worlds, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(a.out / 'prefix_records.csv', index=False)
    meta = dict(block=a.block, root_seed=ROOT_SEED, worlds=len(jobs), records=len(rows),
                seconds=round(time.time() - t0, 1), python=platform.python_version(),
                numpy=np.__version__, platform=platform.platform())
    (a.out / 'run_info.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
