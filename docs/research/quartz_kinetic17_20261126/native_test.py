import subprocess as sp,json,sys,os,pathlib
D=pathlib.Path(__file__).resolve().parent
ROOT=D.parents[2]
MODEL=ROOT/'docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/evidence/mlp_conv_train_20261121/conv_model.json'
MID='161f9e6e26840754384cb47335d1082d837fa014735cbfaa2517209422e29f24'
def request(arm='K17EXP',cap=32,history='4433226223'):
    return dict(cmd='research_connect4_evidence_cpu',history=history,arm=arm,cap=cap,seed=17,model_path=str(MODEL),model_identity=MID)
def main(binary):
    q=request()
    env=dict(os.environ,RAYON_NUM_THREADS='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1')
    p=sp.Popen([binary,'--server'],stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.DEVNULL,text=True,env=env)
    try:
        p.stdin.write(json.dumps(q)+'\n');p.stdin.flush()
        r=json.loads(p.stdout.readline());assert 'error' not in r,r
        e=r['evidence'];a,b=e['pair'];c=e['reservoir'][2];assert e['reservoir'][:2]==[a,b] and c not in [a,b]
        assert [x['target'] for x in e['phases']]==[a,c,None]
        assert e['phases'][1]['n']==24 and e['phases'][2]['n']==32
        ca=[x[0] for x in e['prefix']['candidates']];assert c==next(x for x in ca if x not in [a,b])
        assert e['phases'][1]['delta'][ca.index(c)]==8
        print('PASS actual native third pulse',a,b,c,r['NN_calls'])
    finally:
        p.stdin.close();p.wait(timeout=10)
if __name__=='__main__':main(sys.argv[1])
