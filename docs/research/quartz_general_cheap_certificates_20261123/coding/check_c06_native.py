"""Independent native C06 check; no production generator/verifier imports."""
from pathlib import Path
from fractions import Fraction as Q
import datetime,gzip,hashlib,json,math,struct,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/research/quartz_general_cheap_certificates_20261123'
BIN=ROOT/'target/release/quartz_c24_cheap_certificate'
POLICIES={'PLAIN_STOP','PLAIN_ALWAYS','FAST_ONCE','FAST_BLOCK','OUTWARD_BLOCK','RESIDUAL'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bit(x):return struct.unpack('>Q',struct.pack('>d',x))[0]
def edge_apply(side,rho,x):
    n=len(x);out=[]
    for i,v in enumerate(x):
        degree=(i//side>0)+(i//side+1<side)+(i%side>0)+(i%side+1<side)
        out.append(v*(1+rho*(4-degree)))
    for i in range(n):
        for j in ([i+1] if i%side+1<side else [])+([i+side] if i//side+1<side else []):
            v=rho*(x[i]-x[j]);out[i]+=v;out[j]-=v
    return out
def dot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))
def linear_solve(side,rho,b):
    n=len(b);cols=[edge_apply(side,rho,[Q(int(i==j)) for i in range(n)]) for j in range(n)]
    rows=[[cols[j][i] for j in range(n)]+[b[i]] for i in range(n)]
    assert all(isinstance(x,Q) for row in rows for x in row)
    for k in range(n):
        z=rows[k][k];rows[k]=[v/z for v in rows[k]]
        for i in range(n):
            if i!=k:
                z=rows[i][k];rows[i]=[v-z*w for v,w in zip(rows[i],rows[k])]
    return [row[-1] for row in rows]
def scale_for(bs):
    norm=max(dot(b,b) for b in bs)
    if not norm:return 1.
    s=math.ldexp(1.,math.frexp(float(norm))[1]+1)
    while Q(s)<2*norm:s*=2.
    return s
def make_task(name,side,rho,bs,xstars=None):
    scale=scale_for(bs)
    floats=[[float(v) for v in b] for b in bs]
    assert all(Q(v)==exact for b,bb in zip(floats,bs) for v,exact in zip(b,bb))
    us=[dot(b,x)/Q(scale) for b,x in zip(bs,xstars)] if xstars else [dot(b,linear_solve(side,rho,b))/Q(scale) for b in bs]
    task={'id':name,'side':side,'rho':rho,'scale':scale,'rhs':floats}
    return task,us
tasks=[]
tasks.append(make_task('c24_unit_exact',2,1,[[Q(1),Q(0),Q(0),Q(0)],[Q(0),Q(1,2),Q(1,2),Q(0)],[Q(0)]*4]))
for side,rho,kind in [(32,16,'tie'),(96,64,'wide')]:
    z=[Q(((i*17+(i//side)*3)%31)-15,1024) for i in range(side*side)]
    reflected=[z[i*side+side-1-j] for i in range(side) for j in range(side)]
    xs=[z,reflected,[x/4 for x in z]] if kind=='tie' else [z,[x/1024 for x in reflected],[x/2048 for x in z]]
    bs=[edge_apply(side,rho,x) for x in xs]
    tasks.append(make_task(f'c24_unit_{side}_{kind}',side,rho,bs,xs))
for exponent in [-400,100]:
    a=Q(2)**exponent
    tasks.append(make_task(f'c24_unit_amplitude_{exponent}',2,1,[[a,Q(0),Q(0),Q(0)],[a,a,Q(0),Q(0)],[Q(0)]*4]))
tasks.append(make_task('c24_unit_mixed_subnormal',2,1,[[Q(1),Q(2)**-1074,Q(0),Q(0)],[Q(1,2),Q(0),Q(0),Q(0)],[Q(0)]*4]))
requests=[]
for task,_ in tasks:
    wire=dict(task);wire['scale']=repr(task['scale']);wire['rhs']=[[repr(v) for v in row] for row in task['rhs']]
    requests.append({'task':wire,'additional':32,'repeats':1,'warmup':0,'trace':True})
requests.append({'task':{'id':'c24_known_underflow_reject','side':2,'rho':1,'scale':repr(2.**-1000),'rhs':[[repr(2.**-600),'0','0','0'],['0','0','0','0']]},'additional':8,'repeats':1,'warmup':0,'trace':True})
assert BIN.is_file()
assert not subprocess.check_output(['git','status','--porcelain','--','src/bin/quartz_c24_cheap_certificate.rs'],cwd=ROOT,text=True).strip()
identities={str(BIN.relative_to(ROOT)):sha(BIN),'src/bin/quartz_c24_cheap_certificate.rs':sha(ROOT/'src/bin/quartz_c24_cheap_certificate.rs')}
started=time.perf_counter()
payload=''.join(json.dumps(x,separators=(',',':'))+'\n' for x in requests)
proc=subprocess.run([str(BIN)],input=payload,text=True,capture_output=True)
rawin=DOC/'evidence/C06_NATIVE_UNIT_INPUT.jsonl.gz';rawout=DOC/'evidence/C06_NATIVE_UNIT_OUTPUT.jsonl.gz'
assert not rawin.exists() and not rawout.exists()
for p,data in [(rawin,payload),(rawout,proc.stdout)]:
    with p.open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:z.write(data.encode())
assert proc.returncode==0,proc.stderr
responses=[json.loads(x) for x in proc.stdout.splitlines()]
assert len(responses)==len(requests)
counts={'tasks':len(tasks),'policy_executions':0,'input_components_bit_checked':0,'interval_scalar_containments':0,'nontrivial_scalar_containments':0,'exact_final_residual_components':0,'certified_optimal':0,'identical_noncertified_fallbacks':0,'actual_early_exits':0,'certified_residual_upper_checks':0,'plain_diagnostic_null_checks':0}
def check_bounds(bounds,us):
    assert len(bounds)==len(us)
    for bound,q in zip(bounds,us):
        assert Q(bound['lo'])<=q<=Q(bound['hi'])
        counts['interval_scalar_containments']+=1
        if bound['lo']>0 or bound['hi']<1:counts['nontrivial_scalar_containments']+=1
summaries=[]
for (task,us),response in zip(tasks,responses):
    assert 'error' not in response,response
    assert response['id']==task['id']
    expectedbits=[[bit(v) for v in row] for row in task['rhs']]
    assert response['input_rhs_bits']==expectedbits and response['input_scale_bits']==bit(task['scale'])
    counts['input_components_bit_checked']+=sum(map(len,task['rhs']))
    outs={x['policy']:x for x in response['results']}
    assert set(outs)==POLICIES and len(response['results'])==6
    baseline=outs['PLAIN_ALWAYS']
    if 'tie' in task['id']:assert us[0]==us[1]==max(us)
    for name,out in outs.items():
        counts['policy_executions']+=1
        assert out['solver_matvecs']==sum(out['iterations'])
        assert out['bound_evaluations']==out['fast_evaluations']+out['outward_evaluations']
        for key in ['initial_bounds','final_bounds','decision_bounds']:check_bounds(out[key],us)
        for cp in out['trace']:
            check_bounds(cp['fresh_bounds'],us);check_bounds(cp['bounds'],us)
        if out['certified']:
            assert us[out['selected']]==max(us)
            counts['certified_optimal']+=1
        if name in {'FAST_ONCE','FAST_BLOCK','OUTWARD_BLOCK'}:
            assert us[out['selected']]>=us[baseline['selected']]
            if not out['certified']:
                assert out['selected']==baseline['selected'] and out['iterations']==baseline['iterations']
                assert [[bit(v) for v in row] for row in out['final_x']]==[[bit(v) for v in row] for row in baseline['final_x']]
                assert [bit(b['ell_hat']) for b in out['final_bounds']]==[bit(b['ell_hat']) for b in baseline['final_bounds']]
                counts['identical_noncertified_fallbacks']+=1
            if out['solver_matvecs']<baseline['solver_matvecs']:counts['actual_early_exits']+=1
        if name in {'PLAIN_STOP','PLAIN_ALWAYS'} or (name=='FAST_ONCE' and not out['certified']):
            assert out['residual_upper'] is None and out['residual_goal_met'] is None
            counts['plain_diagnostic_null_checks']+=1
        else:
            uppers=out['residual_upper'];assert len(uppers)==len(us)
            for b,x,upper in zip(task['rhs'],out['final_x'],uppers):
                bf=list(map(Q,b));xf=list(map(Q,x));ax=edge_apply(task['side'],task['rho'],xf)
                res=[v-y for v,y in zip(bf,ax)];D=dot(res,res);bn=dot(bf,bf)
                if bn:assert D<=Q(upper)**2*bn,(task['id'],name)
                else:assert D==0 and upper==0
                counts['exact_final_residual_components']+=len(b)
                counts['certified_residual_upper_checks']+=1
            assert out['residual_goal_met']==all(v<=1e-8 for v in uppers)
    summaries.append({'id':task['id'],'exact_utilities':[str(v) for v in us],'certified':[x for x,o in outs.items() if o['certified']],'matvecs':{x:o['solver_matvecs'] for x,o in outs.items()}})
tiny=responses[-1]
assert 'error' in tiny and tiny['error'] and 'results' not in tiny,tiny
assert counts['certified_optimal']>0 and counts['actual_early_exits']>0 and counts['identical_noncertified_fallbacks']>0
after={str(BIN.relative_to(ROOT)):sha(BIN),'src/bin/quartz_c24_cheap_certificate.rs':sha(ROOT/'src/bin/quartz_c24_cheap_certificate.rs')}
assert after==identities
result={'status':'VERIFIED_NATIVE_FINITE_ASSERTIONS','assertion_tested':'Actual binary parsed bit identity, exact q interval containment, optimal strict certificates, unchanged uncertified continuation, honest nullable plain diagnostics, exact fresh relative residual upper and explicit known-tiny-input rejection.','counts':counts,'cases':summaries,'known_tiny_input_response':tiny,'binary_argv':[str(BIN)],'returncode':proc.returncode,'python_executable':sys.executable,'python':sys.version,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'identities_before':identities,'identities_after':after,'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [rawin,rawout]},'elapsed_seconds':time.perf_counter()-started,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(DOC/'evidence/C06_NATIVE_UNIT_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
