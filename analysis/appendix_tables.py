"""Supplementary tables: calibration by deployment and all stopping rules."""
from common import *
df = pd.read_csv(OUT / 'calibration_bootstrap.csv')
d = df[(df.scope == 'design') & (~df.robust) & (df.scenario == 'mixture') & (df.design != 'all')]
per = d.groupby(['design', 'kind', 'method'])[['viol64', 'viol_any', 'georatio64']].mean()
rows = []
for des in DESIGN_ORDER:
    r = [DESIGN_LABEL[des]]
    for kind, ms in (('point', ('grouped', 'spectral', 'combined')), ('bonf', ('grouped', 'spectral')),
                     ('joint', ('grouped', 'spectral', 'combined'))):
        for m in ms:
            r.append(fmt(per.loc[(des, kind, m)].georatio64, 3))
    rows.append(r)
for kind in ('bonf', 'joint'):
    for m in ('grouped', 'spectral'):
        NUMBERS[f'bydesign_maxviol_{kind}_{m}'] = float(max(per.loc[(des, kind, m)].viol_any for des in DESIGN_ORDER))
NUMBERS['bydesign_bonf_spectral_wins'] = int(sum(per.loc[(des, 'bonf', 'spectral')].georatio64 > per.loc[(des, 'bonf', 'grouped')].georatio64 for des in DESIGN_ORDER))
save_numbers()
lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
         r' & \multicolumn{3}{c}{Single look} & \multicolumn{2}{c}{All looks, Bonferroni} & \multicolumn{3}{c}{All looks, maximum} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}',
         r'Deployment & Grouped & Centred & Comb. & Grouped & Centred & Grouped & Centred & Comb. \\', r'\midrule']
lines += [' & '.join(r) + r' \\' for r in rows] + [r'\bottomrule', r'\end{tabular}']
(TAB / 'calibration_by_design.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
s = pd.read_csv(OUT / 'stopping_summary.csv')
groups = []
for d_ in (0.2, 0.1):
    rows = []
    for (nr, g), sub in s[s.groups >= 10].groupby(['n_robots', 'groups']):
        r = [str(nr), str(g)]
        for rule in ('mixture', 'fixed', 'hoeffding'):
            for H in (8, 1):
                x = sub[(sub.d.round(2) == d_) & (sub.rule == rule) & (sub.H == H)]
                r.append(fmt(float(x.correct.iloc[0]), 2))
        rows.append(r)
    groups.append((f'Margin $d={d_:.2f}$', rows))
write_grouped('stopping_all', ['$N$', '$G$', 'Mix.\\ 8', 'Mix.\\ 1', 'Fixed 8', 'Fixed 1', 'Hoeff.\\ 8', 'Hoeff.\\ 1'],
              groups, 'rrrrrrrr')
print('appendix tables done')
