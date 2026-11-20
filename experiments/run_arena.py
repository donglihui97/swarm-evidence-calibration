"""Bounded-arena validation blocks.

  python experiments/run_arena.py arena        --out results/arena
  python experiments/run_arena.py arena_robust --out results/arena_robust

Each world seed comes from numpy SeedSequence with the fixed root seed and the
world's indices. For every world the script records the pooled swarm estimate
(all readings) and the local estimates of five robots that pool only their
own readings and the broadcasts they received.
"""
from __future__ import annotations
import argparse, json, sys, time, pathlib, platform
from dataclasses import replace, asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarm_ess import LOOKS, arena, estimators as est

ROOT_SEED = 20261119
GRID = 128                      # estimator lattice over the arena side
FLOORS = ('random', 'cluster2', 'cluster4')
FILLS = (0.30, 0.48, 0.52, 0.70)
LOCAL_ROBOTS = 5


def seed_for(*idx):
    return int(np.random.SeedSequence([ROOT_SEED, *idx]).generate_state(1, dtype=np.uint64)[0])


def jobs_arena(worlds=100):
    out = []
    for ki, kind in enumerate(FLOORS):
        for fi, fill in enumerate(FILLS):
            for li, launch in enumerate(('uniform', 'nest')):
                for w in range(worlds):
                    out.append(dict(block='arena', variant='base', floor=kind, fill=fill, launch=launch,
                                    world=w, seed=seed_for(4, ki, fi, li, w), cfg={'launch': launch}))
    return out


ROBUST = {  # variant: ArenaConfig overrides
    'flip05': {'flip': 0.05}, 'flip10': {'flip': 0.10},
    'radio5': {'position': 'radio', 'radio_sd': 0.05}, 'radio10': {'position': 'radio', 'radio_sd': 0.10},
    'odometry': {'position': 'odometry'}, 'N50': {'n_robots': 50},
    'slow': {'speed': 0.02}, 'tile5': {'tile': 0.05},
}


def jobs_robust(worlds=50):
    out = []
    for vi, (v, over) in enumerate(ROBUST.items()):
        for ki, kind in enumerate(FLOORS):
            for fi, fill in enumerate((0.30, 0.48)):
                for li, launch in enumerate(('uniform', 'nest')):
                    for w in range(worlds):
                        out.append(dict(block='arena_robust', variant=v, floor=kind, fill=fill, launch=launch,
                                        world=w, seed=seed_for(5, vi, ki, fi, li, w),
                                        cfg=dict(over, launch=launch)))
    return out


def _grid(x, side):
    return np.clip(x / side * GRID, 0.0, GRID - 1e-9)


def run_world(job):
    cfg = arena.ArenaConfig(**job['cfg'])
    run = arena.simulate_arena(job['seed'], job['floor'], job['fill'], cfg)
    n_t = int(round(cfg.side / cfg.tile))
    tiles = np.minimum((run.x_true / cfg.tile).astype(int), n_t - 1)
    xg = _grid(run.x_rep, cfg.side)
    base = {k: v for k, v in job.items() if k != 'cfg'}
    base.update(n_robots=cfg.n_robots, flip=cfg.flip, f_arena=run.f_arena, target=run.target,
                contacts=run.contacts)
    if cfg.position != 'exact':
        err = np.hypot(*(run.x_rep - run.x_true).transpose(2, 0, 1))
    rows = []
    for t in LOOKS:
        yp, xp = run.y[:, :t], xg[:, :t]
        n = yp.size
        loc = est.pooled_local_ess(yp)
        sp = est.spectral_ess(yp, xp, GRID, loc, centered=True, flip_prob=cfg.flip)
        ref, _ = arena.arena_reference(job['floor'], job['fill'], tiles[:, :t], n_t, cfg.flip)
        row = dict(base, pool='swarm', robot=-1, prefix=t, n=n, success=int(yp.sum()),
                   naive=float(n), additive=float(min(n, loc.sum())), grouped=sp['grouped'],
                   spectral=sp['ess'], reference=ref, amp=sp['amplitude'], length=sp['length'],
                   status=sp['status'], sources=cfg.n_robots)
        if cfg.position != 'exact':
            row['pos_err_median'] = float(np.median(err[:, t - 1]))
        rows.append(row)
        for i in range(LOCAL_ROBOTS):
            m = run.recv[i, :, :t]
            src = np.flatnonzero(m.any(1))
            mk = m[src].astype(float)
            ys = run.y[src, :t]
            xs = xg[src, :t]
            cnt = mk.sum(1)
            nl = int(cnt.sum())
            locl = est.pooled_local_ess(ys, mask=mk)
            spl = est.spectral_ess(ys, xs, GRID, locl, centered=True, flip_prob=cfg.flip, mask=mk)
            th = tiles[src, :t][mk > 0]
            refl, _ = arena.arena_reference(job['floor'], job['fill'], th, n_t, cfg.flip)
            rows.append(dict(base, pool='local', robot=i, prefix=t, n=nl, success=int((ys * mk).sum()),
                             naive=float(nl), additive=float(min(nl, locl.sum())), grouped=spl['grouped'],
                             spectral=spl['ess'], reference=refl, amp=spl['amplitude'],
                             length=spl['length'], status=spl['status'], sources=len(src)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('block', choices=['arena', 'arena_robust'])
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--worlds', type=int, default=None)
    a = ap.parse_args()
    fn = {'arena': jobs_arena, 'arena_robust': jobs_robust}[a.block]
    jobs = fn() if a.worlds is None else fn(a.worlds)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(run_world, jobs, chunksize=2), 1):
            rows += r
            if i % 200 == 0:
                print(f'{a.block}: {i}/{len(jobs)} worlds, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(a.out / 'prefix_records.csv', index=False)
    meta = dict(block=a.block, root_seed=ROOT_SEED, worlds=len(jobs), records=len(rows),
                seconds=round(time.time() - t0, 1), config=asdict(arena.ArenaConfig()),
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform())
    (a.out / 'run_info.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ('block', 'worlds', 'records', 'seconds')}))


if __name__ == '__main__':
    main()
