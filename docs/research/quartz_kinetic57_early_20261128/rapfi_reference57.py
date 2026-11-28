"""Offline initial qualified K56 roots: exact actual Q96/T parity and Q128 comparison."""
import collections as C,json,pathlib
from k57_plan import D,T,sha,load
from native57 import rapfi_module,rsearch,BIN
def main():
    d=D.parent/'quartz_kinetic56_voc_20261128';p=d/'RAPFI_PLAN.json';lk=json.loads(p.read_text())
    e=rapfi_module();ct=C.Counter();out=[]
    with e.Server(BIN) as server:
        for j in lk['jobs']:
            h=j['opening'];seed=e.seed(j,len(h));a=rsearch(e,server,h,seed,True,96);b=rsearch(e,server,h,seed,True,'T');c=rsearch(e,server,h,seed,True,128)
            assert 'target' in b and b['target']['incumbent']==a['move'] and b['votes']==a['votes'] and b['root_nn']==a['nn']
            t=b['target'];ct['roots']+=1;ct['nn96']+=a['nn'];ct['nnT']+=b['nn'];ct['nn128']+=c['nn']
            ct['target_changed']+=b['move']!=a['move'];ct['global_changed']+=c['move']!=a['move'];ct['target_agrees128']+=b['move']==c['move']
            ct['base_agrees128']+=a['move']==c['move'];ct['both_changed']+=b['move']!=a['move'] and c['move']!=a['move']
            out.append(dict(job=j,root96=a,target=b,root128=c))
    assert ct['roots']==72
    (T/'RAPFI_REFERENCE.json').write_text(json.dumps(dict(counts=dict(ct),rows=out,source_lock=sha(p),binary=sha(BIN),scope='offline actual root calls only; Q128 decision not minimax or independent game outcome'),separators=(',',':'),allow_nan=False)+'\n')
    print('RAPFI_REFERENCE',dict(ct),flush=True)
if __name__=='__main__':main()
