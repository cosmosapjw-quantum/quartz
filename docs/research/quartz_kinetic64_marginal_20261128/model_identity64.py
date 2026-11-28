from plan64 import D,T,read,sha
import json,hashlib,torch
def main():
    p=read(D/'PLAN.json');out={}
    for seed,m in sorted(p['models'].items()):
        assert sha(m['path'])==m['sha256']
        raw=torch.load(m['path'],map_location='cpu',weights_only=True);state=raw.get('model_state_dict',raw);h=hashlib.sha256()
        for k,t in sorted(state.items()):
            v=t.detach().cpu().contiguous();h.update(json.dumps([k,str(v.dtype),list(v.shape)],separators=(',',':')).encode()+b'\0');h.update(v.numpy().tobytes())
        out[seed]=dict(tensor_state_sha256=h.hexdigest(),tensor_count=len(state),file_sha256=m['sha256'])
    assert len({z['tensor_state_sha256'] for z in out.values()})==13
    (T/'MODEL_IDENTITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print('13 distinct tensor-state hashes',flush=True)
if __name__=='__main__':main()
