"""Scientific figure generated only from preserved C23 parent summaries."""
import json,pathlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
doc=pathlib.Path(__file__).resolve().parents[1];ev=doc/'evidence'
data=json.loads((ev/'C23_TEST_SUMMARY.json').read_text())
ss=data['summaries'];primary=next(s for s in ss if s['additional']==data['primary_additional'])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
colors={'PLAIN_ALWAYS':'#35465e','CERT_ONCE':'#148b80','CERT_BLOCK':'#e58936','RESIDUAL':'#8659a1','NEVER':'#a3a8ae'}
ax=axes[0,0]
for p in ['CERT_ONCE','CERT_BLOCK']:
 ax.plot([s['additional'] for s in ss],[s['policies']['PLAIN_ALWAYS']['mean_parent_median_us']/s['policies'][p]['mean_parent_median_us'] for s in ss],marker='o',label=p,color=colors[p])
ax.axhline(1,color='black',lw=.8,ls='--');ax.set_xscale('log',base=2);ax.set_xticks([8,32,128],[8,32,128]);ax.set_xlabel('Additional CG cap per action');ax.set_ylabel('PLAIN_ALWAYS time / gate time');ax.set_title('A. End-to-end speed ratio (higher is faster)');ax.legend()
ax=axes[0,1];rows=[r for r in primary['subgroups'] if r['field']=='side'];x=np.arange(len(rows));names=['PLAIN_ALWAYS','CERT_ONCE','CERT_BLOCK','RESIDUAL']
for i,p in enumerate(names):
 ax.bar(x+(i-1.5)*.19,[r['policies'][p]['mean_parent_median_us'] for r in rows],.18,color=colors[p],label=p)
ax.set_yscale('log');ax.set_xticks(x,[str(r['value']) for r in rows]);ax.set_xlabel('Grid side (m); unknowns=m squared');ax.set_ylabel('Mean parent-median elapsed time (us)');ax.set_title('B. Primary cap32 by problem size')
ax=axes[1,0];x=np.arange(len(ss));names=['NEVER','PLAIN_ALWAYS','CERT_ONCE','CERT_BLOCK','RESIDUAL']
for i,p in enumerate(names):
 ax.bar(x+(i-2)*.16,[s['policies'][p]['wrong_decision_parents'] for s in ss],.15,color=colors[p],label=p)
ax.set_xticks(x,[s['additional'] for s in ss]);ax.set_xlabel('Additional CG cap per action');ax.set_ylabel('Parents with positive exact regret /72');ax.set_title('C. Decision errors against exact rational utility');ax.legend(fontsize=8)
ax=axes[1,1];rows=[r for r in primary['subgroups'] if r['field']=='gap'];x=np.arange(len(rows))
for i,p in enumerate(['CERT_ONCE','CERT_BLOCK']):
 values=[r['policies']['PLAIN_ALWAYS']['mean_parent_median_us']-r['policies'][p]['mean_parent_median_us'] for r in rows]
 ax.bar(x+(i-.5)*.33,values,.31,color=colors[p],label=p)
ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[r['value'] for r in rows]);ax.set_ylabel('Time saved vs PLAIN_ALWAYS (us)');ax.set_title('D. Primary cap32 by target utility gap');ax.legend()
fig.suptitle('Quartz C23: useful solver computation, exact decision certificates, full measured cost',fontsize=14)
fig.savefig(ev/'C23_COST_QUALITY.png',dpi=180)
print(str(ev/'C23_COST_QUALITY.png'))
