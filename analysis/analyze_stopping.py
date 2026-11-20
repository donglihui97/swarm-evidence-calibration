"""Certified stopping with independent launch groups (stopping block)."""
from common import *
import matplotlib.pyplot as plt
from swarm_ess import stopping as stp

style()
df = load('stopping')
ALPHA = .05
recs = []
for (nr, G, xi, fill, w), g in df.groupby(['n_robots', 'groups', 'xi', 'fill', 'world']):
    g = g.sort_values('prefix')
    truth = int(g.target.iloc[0] >= .5)
    Z = [np.array([float(v) for v in s.split(';')]) for s in g.group_means]
    prefixes = g.prefix.values
    for rule, fn in stp.RULES.items():
        scores = [fn(z) for z in Z]
        for H in (8, 1):
            thr = stp.threshold(ALPHA, H)
            looks = range(len(prefixes)) if H == 8 else [len(prefixes) - 1]
            stop = None
            for k in looks:
                s = stp.certified_side(scores[k], thr)
                if s is not None:
                    stop = (prefixes[k], s == truth); break
            recs.append(dict(n_robots=nr, groups=G, xi=xi, fill=fill, world=w, rule=rule, H=H,
                             stopped=stop is not None, correct=bool(stop and stop[1]),
                             false=bool(stop and not stop[1]), time=stop[0] if stop else 64,
                             dev=abs(Z[-1].mean() - .5)))
st = pd.DataFrame(recs)
st.to_csv(OUT / 'stopping_records.csv', index=False)
st['d'] = (st.fill - .5).abs().round(2)
summ = st.groupby(['n_robots', 'groups', 'd', 'rule', 'H'])[['correct', 'false', 'time']].mean().reset_index()
summ.to_csv(OUT / 'stopping_summary.csv', index=False)
for _, r in summ.iterrows():
    k = f"N{int(r.n_robots)}_G{int(r.groups)}_d{r.d:.2f}_{r.rule}_H{int(r.H)}"
    NUMBERS[f'stopc_{k}'] = float(r.correct); NUMBERS[f'stopf_{k}'] = float(r.false)
    NUMBERS[f'stopt_{k}'] = float(r.time)
NUMBERS['stop_total_worlds'] = int(st[['n_robots', 'groups', 'xi', 'fill', 'world']].drop_duplicates().shape[0])
NUMBERS['stop_total_false'] = int(st['false'].sum())
NUMBERS['stop_total_decisions'] = int(st.stopped.sum())
NUMBERS['stop_false_by_rule'] = {f'{a}_H{b}': int(v) for (a, b), v in st.groupby(['rule', 'H'])['false'].sum().items()}
NUMBERS['stop_stops_by_rule'] = {f'{a}_H{b}': int(v) for (a, b), v in st.groupby(['rule', 'H'])['stopped'].sum().items()}

# bounds table
rows = []
for d in (.20, .10, .05, .02):
    r = [fmt(d, 2)]
    for H in (8, 1):
        delta = ALPHA / (2 * H)
        gmin = stp.min_groups_constant_vector(d, delta)
        gmix = stp.mixture_min_groups(d, delta)
        ghoef = int(np.ceil(np.log(1 / delta) / (2 * d * d)))
        r += [str(gmin), str(gmix), str(ghoef)]
        NUMBERS[f'gmin_d{d:.2f}_H{H}'] = gmin; NUMBERS[f'gmix_d{d:.2f}_H{H}'] = gmix
        NUMBERS[f'ghoef_d{d:.2f}_H{H}'] = ghoef
    rows.append(r)
write_table('group_bounds', ['$d$', 'Any test', 'Mixture', 'Hoeffding', 'Any test', 'Mixture', 'Hoeffding'],
            rows, 'lrrrrrr')

# figure
from matplotlib.lines import Line2D
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.75), sharey=True)
for ax, d in zip(axes, (.20, .10, .05, .02)):
    for rule, H, col, ls in [('mixture', 8, COL['spectral'], '-'), ('mixture', 1, COL['spectral'], '--'),
                             ('hoeffding', 1, COL['grouped'], '--')]:
        for nr, mk, filled in [(50, 'o', True), (100, 's', False)]:
            g = summ[(summ.n_robots == nr) & (summ.d.round(2) == d) & (summ.rule == rule) & (summ.H == H)]
            ax.plot(g.groups, g.correct, ls, marker=mk, ms=3.5, color=col, mfc=col if filled else 'white', lw=1.3)
    ax.axvline(stp.min_groups_constant_vector(d, ALPHA / 16), color=COL['reference'], ls=':', lw=1)
    ax.axvline(stp.min_groups_constant_vector(d, ALPHA / 2), color=COL['reference'], ls='-.', lw=.8)
    ax.set_xscale('log'); ax.set_xlim(.8, 200)
    ax.set_xticks([1, 2, 5, 10, 25, 50, 100]); ax.set_xticklabels(['1', '2', '5', '10', '25', '50', '100'])
    ax.minorticks_off(); ax.set_title(f'margin $d={d:.2f}$'); ax.set_xlabel('Launch groups $G$')
axes[0].set_ylabel('Correct certified stop')
h = [Line2D([], [], color=COL['spectral'], ls='-', label='Mixture, 8 looks'),
     Line2D([], [], color=COL['spectral'], ls='--', label='Mixture, final look'),
     Line2D([], [], color=COL['grouped'], ls='--', label='Hoeffding, final look'),
     Line2D([], [], color='#52514e', marker='o', ls='', label='50 robots'),
     Line2D([], [], color='#52514e', marker='s', mfc='white', ls='', label='100 robots'),
     Line2D([], [], color=COL['reference'], ls=':', label='$G_{\\min}$, 8 looks'),
     Line2D([], [], color=COL['reference'], ls='-.', label='$G_{\\min}$, final look')]
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(.5, -.01))
fig.tight_layout(rect=(0, .14, 1, 1)); fig.savefig(FIG / 'stopping_groups.png'); fig.savefig(FIG / 'stopping_groups.pdf'); plt.close(fig)
save_numbers()
print('stopping done')
