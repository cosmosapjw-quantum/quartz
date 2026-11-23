"""Render fixed TEST summary; no analysis changes or new experiments."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=Path(__file__).resolve().parents[1];e=d/'evidence';out=d/'figures';out.mkdir(exist_ok=True)
source=e/'C25_TEST_SUMMARY.json';s=json.loads(source.read_text());p=next(x for x in s['summaries'] if x['additional']==32)['policies']
names=['PLAIN_STOP','PLAIN_ALWAYS','FAST_BLOCK','RETAIN_BLOCK','RETAIN_WORK','RETAIN_TIME']
labels=['Initial stop','Fixed budget','Fast certificate','Retain incumbent','Retain + work price','Retain + time estimate']
colors=['#667085','#A6ADBB','#4678B8','#149B86','#D99220','#A14E9E']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','svg.hashsalt':'quartz-c25'})
fig,(a,b)=plt.subplots(1,2,figsize=(14.5,6.5),gridspec_kw={'width_ratios':[1.13,1]},sharey=True)
y=np.arange(6);times=[p[n]['mean_parent_median_us'] for n in names]
a.barh(y,times,color=colors,height=.56,zorder=3)
a.set_yticks(y,labels);a.invert_yaxis();a.set_xlim(0,2030)
for k,n,t in zip(y,names,times):
 a.text(t+24,k,f"{t:,.1f}  |  {p[n]['wrong_decision_parents']}/72",va='center',fontsize=10)
a.set_xlabel('Full native time (microseconds)')
a.set_title('Time and exact decision quality',loc='left',pad=29)
a.text(0,1.025,'Bar label: time | suboptimal decisions',transform=a.transAxes,fontsize=10,color='#475467')
a.grid(axis='x',alpha=.20,zorder=0);a.set_axisbelow(True)
for k,n,col in zip(y,names,colors):
 row=next(x for x in p[n]['priced'] if x['lambda_utility_per_us']==1e-6)
 v=row['mean_net_difference_vs_PLAIN_STOP']*1e6
 lo,hi=np.array(row['paired_parent_bootstrap_net_difference_vs_PLAIN_STOP_95'])*1e6
 b.errorbar(v,k,xerr=np.array([[max(0,v-lo)],[max(0,hi-v)]]),fmt='o',color=col,markersize=7,capsize=4,elinewidth=2,zorder=3)
 b.text(v,k-.22,f'{v:+,.1f}',ha='center',va='center',fontsize=10,color=col,fontweight='bold')
b.axvline(0,color='#344054',lw=1,ls='--');b.set_xlim(-2070,130)
b.set_title('Net utility relative to initial stop',loc='left',pad=29)
b.text(0,1.025,'Point estimate and 95% paired-source bootstrap',transform=b.transAxes,fontsize=10,color='#475467')
b.set_xlabel('Net utility difference (millionths of utility)')
b.grid(axis='x',alpha=.20);b.set_axisbelow(True);b.tick_params(axis='y',left=False,labelleft=False)
fig.suptitle('Quartz C25 | Held-out computation trade-offs',x=.05,ha='left',fontsize=18,fontweight='bold',y=.985)
fig.text(.05,.925,'72 fresh TEST sources · 32 additional CG iterations per action · median of 9 timings per source',fontsize=11,color='#475467')
fig.text(.05,.045,'Price = 1e-6 utility per microsecond. Retention never worsened the initial true utility; it can miss later improvements.',fontsize=10,color='#344054')
fig.text(.05,.015,'These finite SPD results test a computation-control mechanism. Original game-quality targets remain open.',fontsize=10,color='#667085')
fig.subplots_adjust(left=.175,right=.985,top=.80,bottom=.15,wspace=.23)
paths=[out/'C25_TEST_DECISIONS.png',out/'C25_TEST_DECISIONS.svg']
for path in paths:
 assert not path.exists();fig.savefig(path,dpi=180,metadata={'Creator':'Quartz C25 reproducible research'} if path.suffix=='.svg' else {'Software':'Quartz C25 reproducible research'})
plt.close(fig)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rec={'status':'RENDERED_FROM_FROZEN_SUMMARY','summary_sha256':sha(source),'renderer_sha256':sha(Path(__file__)),'artifacts':{str(p.relative_to(d)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}}
(e/'C25_FIGURE.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
