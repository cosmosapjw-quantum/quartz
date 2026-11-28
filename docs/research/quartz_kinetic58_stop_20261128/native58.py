"""Native server client; Python is only evaluator bridge for 7x7."""
import json,time,os,select,subprocess as SP,signal,sys,pathlib
from k58_plan import D,T,ROOT,load,sha
BIN=T/'mcts_demo'
OLD=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic40_budget_backups/mcts_demo'
REF=D.parent/'quartz_kinetic46_confirm_20261127'
sys.path.insert(0,str(REF))
from inc46 import probe
from hybrid46 import exact,bundle,replace
def clk(pid):
    import ctypes as C
    c=C.c_int();f=C.CDLL(None).clock_getcpuclockid;f.argtypes=[C.c_int,C.POINTER(C.c_int)];f.restype=C.c_int
    rc=f(pid,C.byref(c));assert rc==0;return c.value
class NNServer:
    def __init__(self,ev=None,bin=None):
        self.ev=ev;self.err=open(T/'nn_server_stderr.log','a')
        self.p=SP.Popen([str(bin or BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=self.err,bufsize=0,start_new_session=True,env=dict(os.environ,QUARTZ_DISABLE_QIPC_SHM='1',RAYON_NUM_THREADS='1'))
        self.buf=b'';self.clock=clk(self.p.pid)
    def send(self,q):
        self.p.stdin.write((json.dumps(q,separators=(',',':'),allow_nan=False)+'\n').encode());self.p.stdin.flush()
    def read(self):
        until=time.monotonic()+45
        while b'\n' not in self.buf:
            dt=until-time.monotonic()
            if dt<=0 or not select.select([self.p.stdout],[],[],dt)[0]:raise RuntimeError('native timeout')
            z=os.read(self.p.stdout.fileno(),65536)
            if not z:raise RuntimeError('native EOF '+str(self.p.poll()))
            self.buf+=z;assert len(self.buf)<16*1024*1024
        line,self.buf=self.buf.split(b'\n',1);return json.loads(line)
    def request(self,q):
        t=time.perf_counter_ns();c=time.process_time_ns();n=time.clock_gettime_ns(self.clock)
        self.send(q);calls=0;ev0=self.ev.rows if self.ev else 0
        while True:
            a=self.read()
            if a.get('type')!='k47_eval':break
            assert self.ev is not None and a['id']==calls+1 and a['p'] in (1,-1)
            assert a['a']&a['b']==0 and a['legal']==[i for i in range(49) if not (a['a']|a['b'])>>i&1]
            pr,v=self.ev(a['a'],a['b'],a['p'],a['legal']);calls+=1
            self.send(dict(cmd='k47_eval_reply',id=a['id'],a=a['a'],b=a['b'],p=a['p'],prior=pr.tolist(),value=v))
        assert 'error' not in a,a
        if self.ev:assert self.ev.rows-ev0==calls
        if 'nn' in a:assert a.get('physical_nn',a['nn'])==calls
        a['cpu_ns']=time.process_time_ns()-c+time.clock_gettime_ns(self.clock)-n
        a['wall_ns']=time.perf_counter_ns()-t
        return a
    def cached_request(self,q,cache=True,target=False):
        if cache:q['cache']=True
        a=self.request(q)
        root=a.pop('root_nn',None)
        if cache:
            z=a['cache_receipt'];assert a['nn']==z['logical'] and a['physical_nn']==z['physical']
            a['actual_nn']=z['physical']
            assert z['logical']==z['physical']+z['hits']
            # Existing analysis schema uses physical_nn as root-only logical count.
            a['physical_nn']=root if root is not None and 'target' in a else a['nn']
        else:a['actual_nn']=a['nn']
        if 'target' in a:
            assert root is not None and a['nn']==root+a['target']['extra_nn']
            a['native_physical_nn']=a['nn'];a['physical_nn']=root
        return a
    def search(self,h,q,trace=False,cap=None,cache=True):
        return self.cached_request(dict(cmd='research_gomoku7_k47',history=h,cap=(20 if q else 32) if cap is None else cap,quartz=q,trace=trace),cache)
    def nested(self,h,atlas=False,mask=None,trace=False,cache=True):
        return self.cached_request(dict(cmd='research_gomoku7_k48',history=h,cap=20,quartz=True,trace=trace,atlas=atlas,mask=mask),cache)
    def target(self,h,trace=False,reuse=False,cache=True):
        return self.cached_request(dict(cmd='research_gomoku7_k53' if reuse else 'research_gomoku7_k52',history=h,cap=20,quartz=True,trace=trace,atlas=False,mask=0),cache,True)
    def select(self,h,table,trace=False,cache=True):
        return self.cached_request(dict(cmd='research_gomoku7_k55',history=h,cap=20,quartz=True,trace=trace,atlas=False,mask=0,table=table),cache,True)
    def voc(self,h,model,trace=False,cache=True):
        return self.cached_request(dict(cmd='research_gomoku7_k56',history=h,cap=20,quartz=True,trace=trace,atlas=False,mask=0,voc=model),cache,True)
    def early(self,h,horizon,op,trace=False):
        return self.cached_request(dict(cmd='research_gomoku7_k57',history=h,cap=20,quartz=True,trace=trace,atlas=False,mask=0,early=dict(horizon=horizon,op=op)),True,True)
    def stable(self,h,trace=False):
        return self.cached_request(dict(cmd='research_gomoku7_k58',history=h,cap=20,quartz=True,trace=trace,atlas=False,mask=0,early=dict(horizon=16,op=4)),True,True)
    def close(self):
        if self.p.poll() is None:
            self.send(dict(cmd='quit'));self.p.stdin.close();assert self.p.wait(timeout=10)==0
        self.err.close()
    def abort(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid,signal.SIGTERM);self.p.wait(timeout=5)
        self.err.close()
    def __enter__(self):return self
    def __exit__(self,t,*a):
        if t is None:self.close()
        else:self.abort()
def rapfi_module():
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'))
    import experiment as e
    e.BIN=BIN
    return e
def rsearch(e,server,h,seed,q,cap=128):
    t=time.perf_counter_ns();c=time.process_time_ns()
    if not q:a=server.search(h,seed,128,False)
    else:
        req=dict(cmd='research_rapfi_k53' if cap=='T' else 'research_rapfi_k52' if cap in (104,112,'AD') else 'research_rapfi_k48',history=h,seed=seed,cap=None if cap=='AD' else 96 if cap=='T' else cap,horizons=[128]*4+[96]*4 if cap=='AD' else None,lock=True,trace=False)
        a=server.request(req);assert 'error' not in a,a
        assert a['history']==h and a['seed']==seed
        if cap=='AD':
            if a['cause']=='PARTIAL47':assert a['cap']==256
            elif a['proof']['gate'] is not None:assert a['cap']==128
            else:assert a['cap']==req['horizons'][a['bin']]
        else:assert a['cap']==(96 if cap=='T' else cap)
        if a['cause']=='PARTIAL47':
            assert a['partial']['cert']==a['best'] and a['NN_calls']==a['visits']==0 and a['proof']['gate'] is None and a['result'] is None
        else:
            # K36 check applies to original native search, not the post-proof move.
            z=dict(a)
            if a['original_move'] is not None:z['best']=a['original_move']
            e.check(z,dict(req,cmd='research_rapfi_k36'))
    gate=a['proof']['gate'];part=a.get('partial');orig=a.get('original_move')
    v=e.votes(a['result']) or []
    r=dict(selected_cap=a['cap'],move=a['best'],nn=a['NN_calls'],visits=a['visits'],kind='CERTIFICATE' if gate is not None else a['cause'],votes=v,gate=gate,certificate_checks=a.get('certificate_checks',0),native=None,partial=part,original_move=orig,replacement=a.get('replacement',False))
    if a['cause']!='PARTIAL47':
        z=dict(a)
        if orig is not None:z['best']=orig
        r['native']=e.small(z)
    if 'target' in a:
        z=a['target'];assert cap=='T' and q and z['chosen']==a['best']
        assert a['physical_nn']==r['nn']+z['extra_nn']
        r['root_nn']=r['nn'];r['nn']=a['physical_nn'];r['target']=z
    r.update(cpu_ns=time.process_time_ns()-c+sum(a['cpu_ns']),wall_ns=time.perf_counter_ns()-t)
    return r
def normalize(z):
    """Only semantic proof fields; native wall time is not C++ CPU time."""
    if not z:return None
    e=z['exact']
    return dict(cold=z['cold']['receipt'],exact={k:v for k,v in e.items() if k not in ('cpu_ns','wall_ns')} if e else None,cert=z['cert'],bounds=z['bounds'])
