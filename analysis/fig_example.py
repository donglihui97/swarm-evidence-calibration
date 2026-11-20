"""Worked example: one field, three deployments, ESS of each estimator."""
from common import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from swarm_ess import simulator, estimators as est

style()
SEED = 424242
designs = [('Independent origins', 50, 2.0, False), ('Shared origin, $\\sigma=2$', 1, 2.0, False),
           ('Following, $\\sigma=2$', 1, 2.0, True)]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
for k, (ax, (title, g, s, fol)) in enumerate(zip(axes, designs)):
    w = simulator.simulate(SEED, 16.0, .48, g, s, fol)
    loc = est.pooled_local_ess(w.y)
    spec = est.spectral_ess(w.y, w.x, 128, loc)
    ref, _ = est.reference_ess(w.field, w.x)
    ax.imshow(w.field.T, origin='lower', cmap=ListedColormap(['#f4f3ef', '#cfd8e3']), extent=(0, 128, 0, 128))
    for i in range(0, 50, 2):
        p = w.x[i]
        jumps = np.flatnonzero(np.abs(np.diff(p, axis=0)).max(1) > 64) + 1
        for seg in np.split(np.arange(len(p)), jumps):
            ax.plot(p[seg, 0], p[seg, 1], lw=.6, color='#2a2a28', alpha=.6)
        ax.plot(*p[0], 'o', ms=1.6, color='#eb6834')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('#c3c2b7')
    ax.set_title(title)
    ax.set_xlabel(f'grouped {spec["grouped"]:.0f}   centred {spec["ess"]:.0f}   reference {ref:.0f}', fontsize=7)
    NUMBERS[f'example_{k}'] = dict(grouped=spec['grouped'], spectral=spec['ess'], reference=ref,
                                   fraction=float(w.y.mean()), fill=float(w.field.mean()))
fig.tight_layout(); fig.savefig(FIG / 'example.png'); fig.savefig(FIG / 'example.pdf'); plt.close(fig)
save_numbers(); print(NUMBERS)
