"""Recompute raw summary consistency and draw figures from the frozen pilot."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
path = HERE/'evidence'/'pilot'
summary = json.loads((path/'summary.json').read_text())
raw = path/'raw.csv.gz'
assert hashlib.sha256(raw.read_bytes()).hexdigest() == summary['raw_sha256']
acc = {}
with gzip.open(raw,'rt') as f:
    for row in csv.DictReader(f):
        key = row['raw_key']
        values = acc.setdefault(key, [0,0.,0,0])
        assert int(row['unit']) == values[0]
        values[0] += 1
        values[1] += float(row['regret'])
        values[2] += int(row['error'])
        values[3] += int(row['covered95'])
for row in summary['rows']:
    count,total,error,covered = acc[row['raw_key']]
    assert count == summary['worlds_per_scenario']
    assert row['logical_evaluations'] == row['budget'] == sum(row['counts'])
    np.testing.assert_allclose(total/count,row['mean_regret'],rtol=1e-8,atol=1e-12)
    assert error/count == row['wrong_action_rate']
    assert covered/count == row['coverage95']
rows = summary['rows']
lookup = {(r['scenario'],r['width'],r['budget'],r['method']):r for r in rows}
report = {'input_summary_sha256':hashlib.sha256((path/'summary.json').read_bytes()).hexdigest(),
          'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'unique_raw_designs':len(acc),'raw_units':sum(x[0] for x in acc.values()),
          'design_rows_verified':len(rows),'exact_high_budget_comparison':[]}
for scenario in ('independent','positive','negative'):
    e,n,s,p = [lookup[(scenario,32,2048,m)] for m in summary['methods']]
    report['exact_high_budget_comparison'].append({'scenario':scenario,
       'pending_vs_equal_relative_regret_reduction':1-p['exact_bayes_regret']/e['exact_bayes_regret'],
       'pending_vs_neyman_relative_regret_reduction':1-p['exact_bayes_regret']/n['exact_bayes_regret'],
       'pending_vs_stale_relative_regret_reduction':1-p['exact_bayes_regret']/s['exact_bayes_regret']})
report['coverage_misspecified_b8_pending']=lookup[('misspecified',32,8,'pending_greedy')]['coverage95']
report['coverage_correct_positive_b8_pending']=lookup[('positive',32,8,'pending_greedy')]['coverage95']
report['interpretation']='exploratory exact Gaussian mechanism; no demonstrated high-budget advantage over known-noise allocation; no NN/MCTS runtime or game result'
out=HERE/'evidence'/'analysis'
out.mkdir(exist_ok=False)
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
styles={'equal':('#747b84','o'),'neyman':('#d07d24','s'),'stale_batch':('#b54f50','^'),'pending_greedy':('#176d91','D')}
labels={'equal':'Equal allocation','neyman':'Known-noise Neyman','stale_batch':'Stale batch','pending_greedy':'Pending covariance'}
for m,(color,marker) in styles.items():
    selected=[lookup[('independent',32,b,m)] for b in summary['budgets']]
    axes[0].plot(summary['budgets'],[r['exact_bayes_regret'] for r in selected],marker=marker,
                 color=color,label=labels[m],lw=1.6,ms=5,alpha=.85)
axes[0].set(xscale='log',yscale='log',xlabel='Completed synthetic observations',ylabel='Exact expected simple regret',title='Two finalists; batch width 32')
axes[0].legend(frameon=False,fontsize=8)
for scenario,label,color in [('positive','Correct correlation model','#176d91'),('misspecified','Wrong-sign correlation model','#b54f50')]:
    selected=[lookup[(scenario,32,b,'pending_greedy')] for b in summary['budgets']]
    axes[1].plot(summary['budgets'],[r['coverage95'] for r in selected],marker='o',label=label,color=color)
axes[1].axhline(.95,color='#747b84',ls='--',lw=1,label='Nominal 95%')
axes[1].set(xscale='log',ylim=(.35,1),xlabel='Completed synthetic observations',ylabel='Prior-world gap interval coverage',title='4096 worlds per scenario; exploratory')
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('QUARTZ diagnostic: allocation gain survives, novelty and calibration do not follow',fontsize=12)
fig.savefig(out/'pilot_budget_and_calibration.png',dpi=160)
fig.savefig(out/'pilot_budget_and_calibration.pdf')
plt.close(fig)
print(json.dumps(report,indent=2))
