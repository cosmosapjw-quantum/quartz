"""C09 bounded independent checks. No production imports; no source mutations."""
import json, math, hashlib, platform, subprocess, sys
from fractions import Fraction as F
from pathlib import Path

ROOT = Path('/home/cosmosapjw/quantum/research/quartz_general_meta23_20261123')
BIN = ROOT/'target/release/quartz_c23_residual_gate'
SOURCE = ROOT/'src/bin/quartz_c23_residual_gate.rs'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def edge_pairs(m):
    return [(i*m+j,i*m+j+1) for i in range(m) for j in range(m-1)] + [(i*m+j,(i+1)*m+j) for i in range(m-1) for j in range(m)]
def op(v,m,rho):
    out=list(v)
    degree=[0]*len(v)
    for i,j in edge_pairs(m):
        z=rho*(v[i]-v[j])
        out[i]+=z; out[j]-=z
        degree[i]+=1;degree[j]+=1
    return [z+rho*(4-d)*v[i] for i,(z,d) in enumerate(zip(out,degree))]
def dense_solve(b,m,rho):
    n=len(b)
    columns=[op([F(i==j) for i in range(n)],m,rho) for j in range(n)]
    a=[[columns[j][i] for j in range(n)]+[b[i]] for i in range(n)]
    for i in range(n):
        pivot=next(j for j in range(i,n) if a[j][i])
        a[i],a[pivot]=a[pivot],a[i]
        c=a[i][i];a[i]=[z/c for z in a[i]]
        for j in range(n):
            if j!=i:
                c=a[j][i];a[j]=[z-c*w for z,w in zip(a[j],a[i])]
    return [row[-1] for row in a]
def energy(x,m,rho):
    degree=[0]*len(x)
    out=sum(z*z for z in x)
    for i,j in edge_pairs(m):
        out+=rho*(x[i]-x[j])**2
        degree[i]+=1;degree[j]+=1
    return out+rho*sum((4-d)*z*z for d,z in zip(degree,x))
def make_task(ident,m,rho,bs,qs):
    norm=max(sum(z*z for z in b) for b in bs)
    scale=F(1)
    while scale<=2*norm:scale*=2
    task={'id':ident,'side':m,'rho':rho,'scale':repr(float(scale)),
          'rhs':[[repr(float(z)) for z in b] for b in bs]}
    assert all(F(float(z))==z for b in bs for z in b)
    return task,[q/scale for q in qs]
def dense_task(ident,m,rho,bs):
    bs=[[F(z) for z in b] for b in bs]
    xs=[dense_solve(b,m,rho) for b in bs]
    qs=[sum(z*w for z,w in zip(b,x)) for b,x in zip(bs,xs)]
    return make_task(ident,m,rho,bs,qs)
def manufactured_task(ident,m,rho,xs):
    xs=[[F(z) for z in x] for x in xs]
    bs=[op(x,m,rho) for x in xs]
    qs=[energy(x,m,rho) for x in xs]
    assert qs==[sum(z*w for z,w in zip(x,b)) for x,b in zip(xs,bs)]
    return make_task(ident,m,rho,bs,qs)

cases=[
    dense_task('c09-dense2',2,1,[[1,0,0,0],[0,F(1,2),F(1,2),0],[F(1,8)]*4]),
    dense_task('c09-dense3',3,16,[[F((i*7)%11-5,8) for i in range(9)],[F((i*5)%13-6,16) for i in range(9)],[F(1,64)]*9]),
    dense_task('c09-decimal',2,1,[[F(1.8920180462300777),F(-10.504335880279541),F(0),F(0)],[F(0)]*4])
]
x=[F((i*17)%23-11,4096) for i in range(64)]
cases.append(manufactured_task('c09-wide8',8,64,[x,[z/8 for z in x],[z/16 for z in x]]))
reflection=[x[i*8+7-j] for i in range(8) for j in range(8)]
cases.append(manufactured_task('c09-tie8',8,16,[x,reflection,[z/8 for z in x]]))
cases.append(manufactured_task('c09-rho0',3,0,[[F(i-4,16) for i in range(9)],[F(0)]*9]))
cases.append(manufactured_task('c09-zero',2,0,[[F(0)]*4,[F(0)]*4]))

before={'binary':sha(BIN),'rust_source':sha(SOURCE)}
counts={'tasks':0,'native_policy_executions':0,'scalar_q_containments':0,
        'exact_final_residual_bound_checks':0,'transport_components':0,
        'certified_optimal':0,'noncertified_full_state_matches':0,
        'plain_full_state_matches':0,'native_exit_codes':[]}
detail=[]
for task,q in cases:
    req={'task':task,'additional':32,'repeats':1,'warmup':0,'trace':True}
    cp=subprocess.run([str(BIN)],input=json.dumps(req)+'\n',capture_output=True,text=True)
    counts['native_exit_codes'].append(cp.returncode)
    assert cp.returncode==0,cp.stderr
    result=json.loads(cp.stdout);assert 'error' not in result,result
    counts['tasks']+=1
    bs=[[F(float(z)) for z in b] for b in task['rhs']]
    scale=F(float(task['scale']));m=task['side'];rho=task['rho']
    import struct
    bits=lambda x:struct.unpack('<Q',struct.pack('<d',x))[0]
    assert result['input_rhs_bits']==[[bits(float(z)) for z in b] for b in task['rhs']]
    assert result['input_scale_bits']==bits(float(task['scale']))
    counts['transport_components']+=sum(map(len,bs))
    policies={r['policy']:r for r in result['results']}
    assert set(policies)=={'NEVER','ALWAYS','PLAIN_ALWAYS','CERT_ONCE','CERT_BLOCK','RESIDUAL'}
    for r in result['results']:
        counts['native_policy_executions']+=1
        families=[r['initial_bounds'],r['final_bounds'],r['decision_bounds']]
        for t in r['trace']:families.extend([t['bounds'],t['fresh_bounds']])
        for family in families:
            for z,bound in zip(q,family):
                assert F(bound['lo'])<=z<=F(bound['hi']),(task['id'],r['policy'],'q containment')
                counts['scalar_q_containments']+=1
        for b,xhat,bound in zip(bs,r['final_x'],r['final_bounds']):
            xh=[F(z) for z in xhat];ax=op(xh,m,rho)
            ell=2*sum(z*w for z,w in zip(b,xh))-sum(z*w for z,w in zip(xh,ax))
            rr=sum((z-w)**2 for z,w in zip(b,ax))
            exact_l=max(F(0),(ell+rr/F(1+8*rho))/scale)
            exact_u=min(F(1),(ell+rr)/scale)
            assert F(bound['lo'])<=exact_l and F(bound['hi'])>=exact_u,(task['id'],r['policy'],'outward bound')
            counts['exact_final_residual_bound_checks']+=1
        if r['certified']:
            assert q[r['selected']]==max(q)
            selected=r['decision_bounds'][r['selected']]['lo']
            assert all(i==r['selected'] or selected>b['hi'] for i,b in enumerate(r['decision_bounds']))
            counts['certified_optimal']+=1
    a=policies['ALWAYS'];plain=policies['PLAIN_ALWAYS']
    assert plain['iterations']==a['iterations'] and plain['final_x']==a['final_x']
    assert plain['selected']==a['selected']
    assert [b['ell_hat'] for b in plain['final_bounds']]==[b['ell_hat'] for b in a['final_bounds']]
    counts['plain_full_state_matches']+=1
    for name in ['CERT_ONCE','CERT_BLOCK']:
        r=policies[name]
        assert q[r['selected']]>=q[a['selected']]
        if not r['certified']:
            assert r['iterations']==a['iterations'] and r['final_x']==a['final_x'] and r['selected']==a['selected']
            assert [b['ell_hat'] for b in r['final_bounds']]==[b['ell_hat'] for b in a['final_bounds']]
            counts['noncertified_full_state_matches']+=1
    if task['id']=='c09-tie8':assert all(not policies[p]['certified'] for p in ['CERT_ONCE','CERT_BLOCK'])
    if task['id']=='c09-wide8':assert policies['CERT_ONCE']['solver_matvecs']<a['solver_matvecs']
    detail.append({'id':task['id'],'oracle_exact':[str(x) for x in q],
                   'policies':{p:{k:r[k] for k in ['selected','certified','iterations','solver_matvecs','residual_goal_met']} for p,r in policies.items()}})

tiny={'task':{'id':'c09-underflow','side':2,'rho':1,'scale':repr(math.ldexp(1.,-1000)),
              'rhs':[[repr(math.ldexp(1.,-600)),'0.0','0.0','0.0'],['0.0']*4]},
      'additional':8,'repeats':1,'warmup':0,'trace':True}
cp=subprocess.run([str(BIN)],input=json.dumps(tiny)+'\n',capture_output=True,text=True)
assert cp.returncode==0,cp.stderr
out=json.loads(cp.stdout);assert 'error' not in out,out
res=next(r for r in out['results'] if r['policy']=='RESIDUAL')
assert res['iterations']==[0,0] and res['residual_goal_met'] and res['residual_relative']==[0.0,0.0]
assert all(v==0.0 for v in res['final_x'][0])
tinyq=[F(23,105)*F(2)**(-200),F(0)]
tiny_checks=0
for rr in out['results']:
    families=[rr['initial_bounds'],rr['final_bounds'],rr['decision_bounds']]
    for t in rr['trace']:families.extend([t['bounds'],t['fresh_bounds']])
    for family in families:
        for qj,bj in zip(tinyq,family):
            assert F(bj['lo'])<=qj<=F(bj['hi'])
            tiny_checks+=1
underflow={'input':tiny,'native_exit_code':cp.returncode,'observed':{k:res[k] for k in ['reason','iterations','residual_relative','residual_goal_met','certified']},
           'exact_nonzero_arm_relative_residual':'1','exact_normalized_utility':str(tinyq[0]),'outward_containment_checks':tiny_checks,'outward_certificate_remained_enclosing':True,'classification':'nonblocking outside manufactured source range; accepted-input diagnostic defect'}
after={'binary':sha(BIN),'rust_source':sha(SOURCE)}
assert before==after
output={'status':'INDEPENDENT_FINITE_CHECKS_PASSED_WITH_ONE_OUT_OF_COHORT_DIAGNOSTIC_DEFECT',
        'assertion_tested':'Frozen native binary on seven independently assembled exact finite tasks; exact response containment and final residual bounds; same-state fallbacks; transport bits. One tiny-input diagnostic reproducer.',
        'argv':['python3','/tmp/quartz_c23_review.py'],'python':sys.version,'platform':platform.platform(),
        'input_hashes_before':before,'input_hashes_after':after,'counts':counts,'details':detail,'nonblocking_finding':underflow}
print(json.dumps(output,indent=2))
