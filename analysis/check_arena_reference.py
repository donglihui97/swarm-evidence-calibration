"""Monte Carlo check of the arena reference (Equation eq:arena-ref)."""
from common import *
from swarm_ess import arena
from experiments_seed import seed_for

dev = []
for ki, kind in enumerate(('random', 'cluster2', 'cluster4')):
    for li, launch in enumerate(('uniform', 'nest')):
        for flip in (0.0, 0.1):
            cfg = arena.ArenaConfig(launch=launch, flip=flip)
            run = arena.simulate_arena(seed_for(9, ki, li), kind, 0.30, cfg)
            tiles = np.minimum((run.x_true / cfg.tile).astype(int), 19)
            _, var = arena.arena_reference(kind, 0.30, tiles, 20, flip)
            rng = np.random.default_rng(seed_for(9, ki, li, 1))
            err = []
            for _ in range(4000):
                fl = arena.make_floor(kind, 0.30, 20, rng)
                yy = fl[tiles[..., 0], tiles[..., 1]]
                if flip:
                    yy = np.where(rng.random(yy.shape) < flip, 1 - yy, yy)
                err.append(yy.mean() - (flip + (1 - 2 * flip) * fl.mean()))
            dev.append(var / np.var(err) - 1)
            print(kind, launch, flip, f'{var / np.var(err):.3f}')
NUMBERS['arena_refcheck_maxdev'] = float(np.max(np.abs(dev)))
NUMBERS['arena_refcheck_cases'] = len(dev)
save_numbers()
print('max deviation', np.max(np.abs(dev)))
