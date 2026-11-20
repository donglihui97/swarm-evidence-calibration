# swarm-evidence-calibration

Code and simulation records for the paper

> Lihui Dong, Zhuo Li and Yue Dong. *How much independent evidence does a swarm collect? Spatial coupling, information fusion and calibrated confidence in collective perception.* Submitted to *Adaptive Behavior*.

A robot swarm that perceives its environment collectively fuses thousands of binary readings into one decision. Readings taken close together in space repeat the same evidence, so the number of readings overstates what the swarm knows. How strongly it does so depends on the swarm's own behaviour: a shared launch site, a leader–follower formation or fusing readings received over short-range radio all make the robots' readings depend on each other. This repository measures that loss and corrects for it.

It contains

- a simulator of binary fields and deployments on a periodic arena, and a second simulator of a walled arena with colliding robots, nest launches, lossy local radio and imperfect positions;
- estimators of the effective sample size (ESS), among them the **centred spectral estimator**, which fits a covariance model to all cross-robot pairs of readings and evaluates every pair sum exactly from per-robot Fourier sums;
- split-conformal calibration of the ESS and betting e-values for certified stopping;
- the raw per-world records of all five experiment blocks (37,400 simulated worlds) and the scripts that turn them into every table and figure of the paper.

## Example

```python
import numpy as np
from swarm_ess import simulator, estimators as est

w = simulator.simulate(seed=1, xi=16.0, fill=0.48, groups=1)  # 50 robots, one shared launch
local = np.array([est.local_ess(s) for s in w.y])              # per-robot temporal correction
counts = np.full(len(w.y), w.y.shape[1])
grouped, _ = est.grouped_local_ess(counts, local)              # temporal correction only
centred = est.spectral_ess(w.y, w.x, 128, local)['ess']        # adds cross-robot dependence
reference, _ = est.reference_ess(w.field, w.x)                 # needs the true field
```

For this world the 3,200 readings are worth 8.0 independent ones according to the reference. The temporal correction alone reports 107.5, and the centred spectral estimator reports 6.4. With 50 independent launches (`groups=50`) the same quantities are 68.1, 147.2 and 115.3.

`y` is an `N x T` array of binary readings and `x` an `N x T x 2` array of positions on an `L x L` periodic arena. For streams with gaps, such as the readings a robot receives from its neighbours, pass a 0/1 array as `mask=`.

## Installation

Python 3.10 or later.

```bash
pip install -e .            # numpy, scipy, pandas, matplotlib
pip install pytest
python -m pytest -q tests   # 13 tests, about 2 s
```

The tests include a check of the exact Fourier pair sums (Proposition 2 of the paper) against direct summation over all pairs.

## Repository layout

| Path | Contents |
|---|---|
| `swarm_ess/simulator.py` | binary fields; independent, grouped, shared and following deployments; persistent random walk; sensor flips |
| `swarm_ess/estimators.py` | local, pooled local and grouped local ESS; centred and uncentred spectral ESS; evaluation-only reference ESS |
| `swarm_ess/calibration.py` | split-conformal quantiles and scale standardisation |
| `swarm_ess/stopping.py` | betting e-values, Hoeffding test, minimum number of launch groups |
| `swarm_ess/beta.py` | Beta credible reports with a given ESS |
| `swarm_ess/arena.py` | walled arena simulator and its exact reference ESS |
| `experiments/` | scripts that run the five experiment blocks |
| `analysis/` | scripts that produce the tables, figures and numbers of the paper |
| `results/<block>/` | raw per-world records (`prefix_records.csv.gz`) and run information |
| `tests/` | unit tests |

## Reproducing the paper

### From the released records

The analysis scripts read `results/<block>/prefix_records.csv.gz` directly and write to `outputs/`.

```bash
cd analysis
python fig_example.py            # example world (Figure 1)
python analyze_bank.py           # ESS accuracy, confidence and posterior-stopping tables and figures
python analyze_calibration.py    # conformal calibration tables and figure
python analyze_stopping.py       # stopping figure and group-bound table
python analyze_robustness.py     # robustness table
python appendix_tables.py        # appendix tables
python analyze_arena.py          # arena tables and figure
python check_arena_reference.py  # Monte Carlo check of the arena reference
python ablation_local.py         # local-radio ablation (runs its own simulations)
python benchmark.py              # timing of the spectral fit against direct summation
```

| Output | Script |
|---|---|
| `ess_accuracy.tex`, `confidence.tex`, `confidence_design.tex`, `posterior_stopping.tex`, `deployment_accuracy.pdf`, `prefix_accuracy.pdf`, `confidence_design.pdf` | `analyze_bank.py` |
| `calibration_main.tex`, `calibration_shift.tex`, `calibration.pdf` | `analyze_calibration.py` |
| `group_bounds.tex`, `stopping_groups.pdf` | `analyze_stopping.py` |
| `robustness.tex` | `analyze_robustness.py` |
| `calibration_by_design.tex`, `stopping_all.tex` | `appendix_tables.py` |
| `arena_accuracy.tex`, `arena_confidence.tex`, `arena_calibration.tex`, `arena_robust.tex`, `arena.pdf` | `analyze_arena.py` |
| `ablation_local.tex` | `ablation_local.py` |
| `example.pdf` | `fig_example.py` |

Numbers quoted in the text are collected in `outputs/numbers.json`.

### From scratch

Each block writes a new `prefix_records.csv`, which takes precedence over the compressed file.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/run_experiments.py bank       --out results/bank
python experiments/run_experiments.py stopping   --out results/stopping
python experiments/run_experiments.py robustness --out results/robustness
python experiments/run_arena.py arena            --out results/arena
python experiments/run_arena.py arena_robust     --out results/arena_robust
```

All blocks, including the ablation, take about 100 minutes on a 24-thread laptop processor (Intel i9-12900HX) with Python 3.14 and NumPy 2.5. `--workers` sets the number of processes and does not change the results; `--worlds` runs a smaller test version of a block.

## Licence

MIT. See `LICENSE`.