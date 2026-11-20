"""Seed derivation shared by the experiment and analysis scripts."""
import numpy as np

ROOT_SEED = 20261119


def seed_for(*idx):
    return int(np.random.SeedSequence([ROOT_SEED, *idx]).generate_state(1, dtype=np.uint64)[0])
