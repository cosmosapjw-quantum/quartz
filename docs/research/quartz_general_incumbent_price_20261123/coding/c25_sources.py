"""Exact dyadic manufactured SPD sources for Quartz C25 (no solver outcomes used).

Public b/scale are functions of preserved source specifications. The hidden
manufactured solution and rational utility are written to separate oracle files.
"""
import argparse,decimal,gzip,hashlib,json,math,pathlib,random
from fractions import Fraction
P = 30
def integer_stencil(v,side,rho):
    out=[]
    for i,x in enumerate(v):
        total=(1+4*rho)*x
        if i%side:total-=rho*v[i-1]
        if i%side+1<side:total-=rho*v[i+1]
        if i>=side:total-=rho*v[i-side]
        if i+side<len(v):total-=rho*v[i+side]
        out.append(total)
    return out
def shape(rng,side):
    modes=[(rng.randint(1,min(side,7)),rng.randint(1,min(side,7)),rng.uniform(-1,1)) for _ in range(8)]
    vals=[]
    for i in range(side):
        for j in range(side):
            value=sum(c*math.sin(math.pi*k*(i+1)/(side+1))*math.sin(math.pi*l*(j+1)/(side+1)) for k,l,c in modes)
            vals.append(round(1024*(value+0.05*rng.uniform(-1,1))))
    if not any(vals):vals[0]=1
    return vals
def make_source(spec):
    side,rho=int(spec['side']),int(spec['rho'])
    if not 2<=side<=128 or not 0<=rho<=64:raise ValueError('unsupported operator')
    rng=random.Random(int(spec['seed']))
    raw=[shape(rng,side) for _ in range(3)]
    if spec['target_gap']=='tie':
        raw[1]=[raw[0][i*side+side-1-j] for i in range(side) for j in range(side)]
    energies=[sum(x*y for x,y in zip(v,integer_stencil(v,side,rho))) for v in raw]
    ratio={'wide':'0.75','medium':'0.97','tight':'0.99999','tie':'1'}[spec['target_gap']]
    targets=['1',ratio,'0.60']
    vectors=[]
    with decimal.localcontext() as ctx:
        ctx.prec=60
        for index,(v,q,target) in enumerate(zip(raw,energies,targets)):
            if index==0 or (index==1 and spec['target_gap']=='tie'):
                factor=1<<20
            else:
                factor=int((decimal.Decimal(target)*decimal.Decimal(energies[0])/decimal.Decimal(q)).sqrt()*(1<<20)+decimal.Decimal('0.5'))
            vectors.append([x*factor for x in v])
    order=list(range(3));rng.shuffle(order)
    vectors=[vectors[i] for i in order]
    bs=[integer_stencil(v,side,rho) for v in vectors]
    max_abs=max(abs(x) for row in bs+vectors for x in row)
    assert max_abs < 2**53, 'dyadic conversion is not exact'
    max_norm_num=max(sum(x*x for x in row) for row in bs)
    # One extra binary exponent ensures room for outward norm validation.
    exponent=max_norm_num.bit_length()-2*P+1
    scale=math.ldexp(1.0,exponent)
    rhs=[[math.ldexp(float(x),-P) for x in row] for row in bs]
    q=[Fraction(sum(x*y for x,y in zip(v,b)),1<<(2*P))/Fraction(scale) for v,b in zip(vectors,bs)]
    for b in bs:assert Fraction(sum(x*x for x in b),1<<(2*P))<=Fraction(scale)
    assert all(Fraction(0)<=x<=Fraction(1) for x in q)
    task={'id':spec['id'],'side':side,'rho':rho,'scale':scale,'rhs':rhs}
    sorted_q=sorted(q,reverse=True)
    oracle={'id':spec['id'],'spec':spec,'dyadic_exponent':P,'x_num':vectors,'u_exact':[str(x) for x in q],'optimal_actions':[i for i,x in enumerate(q) if x==max(q)],'exact_gap':str(sorted_q[0]-sorted_q[1]),'max_abs_integer':max_abs}
    return task,oracle
def write_gzip(path,rows):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as stream:
            for row in rows:stream.write((json.dumps(row,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n').encode())

# The manufacture functions above are reused without changing their arithmetic.
UNIT_SPECS = [
    {"id":"c25_unit_00","split":"UNIT","side":2,"rho":0,"target_gap":"wide","seed":25090001,"stratum":0},
    {"id":"c25_unit_01","split":"UNIT","side":3,"rho":1,"target_gap":"tie","seed":25090002,"stratum":1},
    {"id":"c25_unit_02","split":"UNIT","side":4,"rho":16,"target_gap":"tight","seed":25090003,"stratum":2},
    {"id":"c25_unit_03","split":"UNIT","side":8,"rho":64,"target_gap":"medium","seed":25090004,"stratum":3},
]
UNIT_SPECS.append({'id': 'c24_test_34_1', 'rho': 64, 'seed': 24034944, 'side': 96, 'split': 'TEST', 'stratum': 34, 'target_gap': 'tight'})  # Regression only; never a fresh cohort parent.

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def configured_specs(config, split):
    return UNIT_SPECS if split=="UNIT" else [s for s in config["sources"] if s["split"]==split]

def validate_configuration(config, previous=None):
    from collections import Counter
    specs=config["sources"]
    assert len(specs)==config["source_count"]==144
    assert len({s["id"] for s in specs})==len({s["seed"] for s in specs})==144
    assert Counter(s["split"] for s in specs)=={"TRAIN":36,"CAL":36,"TEST":72}
    assert set(s["stratum"] for s in specs)==set(range(36))
    for stratum in range(36):
        ss=[s for s in specs if s["stratum"]==stratum]
        assert Counter(s["split"] for s in ss)=={"TRAIN":1,"CAL":1,"TEST":2}
        assert len({(s["side"],s["rho"],s["target_gap"]) for s in ss})==1
    assert len({(s["side"],s["rho"],s["target_gap"]) for s in specs})==36
    assert config["additional"]==[8,32,128]
    assert config["repeats"]==9 and config["warmup"]==1
    assert config["policies"]==["PLAIN_STOP","PLAIN_ALWAYS","FAST_BLOCK","RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME"]
    assert config["prices"]==[0,1e-8,1e-7,1e-6,1e-5]
    assert config["bootstrap_seed"]==2500923 and config["bootstrap_resamples"]==2000
    assert config["primary_policy"]=="RETAIN_TIME" and config["primary_baseline"]=="PLAIN_STOP"
    assert config["primary_additional"]==32 and Fraction(config["primary_price_exact"])==Fraction(1,1000000)
    assert Fraction(config["work_price_exact"])==Fraction(1,1<<30)
    assert config["predictor"]["fit_split"]=="TRAIN" and config["predictor"]["fit_policy"]=="RETAIN_BLOCK"
    assert config["predictor"]["fit_additional"]==32
    assert len(config["residual_exact_subset"])==len(set(config["residual_exact_subset"]))==9
    assert set(config["residual_exact_subset"])<=set(s["id"] for s in specs if s["split"]=="TEST")
    if previous:
        old={s["id"]:s for s in previous["sources"]}
        for spec in specs:
            source=old[spec["id"].replace("c25_","c24_",1)]
            assert spec["seed"]==source["seed"]+1000000
            assert {k:v for k,v in spec.items() if k not in ("id","seed")}=={k:v for k,v in source.items() if k not in ("id","seed")}
        assert {s["id"] for s in specs}.isdisjoint(s["id"] for s in previous["sources"])
        assert {s["seed"] for s in specs}.isdisjoint(s["seed"] for s in previous["sources"])

def main():
    import datetime,platform,subprocess,sys
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],required=True)
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1];root=doc.parents[2]
    config=json.loads((doc/"SOURCE_SPECS.json").read_text())
    previous=root/"docs/research/quartz_general_cheap_certificates_20261123/SOURCE_SPECS.json"
    validate_configuration(config,json.loads(previous.read_text()) if previous.exists() else None)
    specs=configured_specs(config,args.split)
    ev=args.evidence_dir or doc/"evidence";ev.mkdir(parents=True,exist_ok=True)
    public=ev/f"C25_PUBLIC_{args.split}.jsonl.gz"
    oracle=ev/f"C25_ORACLE_{args.split}.jsonl.gz"
    manifest=ev/f"C25_SOURCES_{args.split}.json"
    if any(p.exists() for p in [public,oracle,manifest]):raise FileExistsError("source output exists; preserve and inspect")
    inputs={str(p.relative_to(root)):sha(p) for p in [doc/"SOURCE_SPECS.json",pathlib.Path(__file__)]}
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],cwd=root,text=True).splitlines()
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Manufacture and write one parent at a time; hidden labels are never merged into public task.
    with public.open("xb") as pf,oracle.open("xb") as of:
        with gzip.GzipFile(filename="",fileobj=pf,mode="wb",mtime=0) as pg,gzip.GzipFile(filename="",fileobj=of,mode="wb",mtime=0) as og:
            for spec in specs:
                task,truth=make_source(spec)
                for dest,row in [(pg,task),(og,truth)]:
                    dest.write((json.dumps(row,separators=(",",":"),sort_keys=True,allow_nan=False)+"\n").encode())
                print(json.dumps({"source_generated":spec["id"]}),flush=True)
    assert all(sha(root/p)==h for p,h in inputs.items()),"generation inputs changed"
    result={"schema":"quartz.c25.sources.v1","status":"completed","split":args.split,
        "count":len(specs),"source_ids":[s["id"] for s in specs],"source_specs_sha256":sha(doc/"SOURCE_SPECS.json"),
        "generator_sha256":sha(pathlib.Path(__file__)),"input_hashes_before":inputs,"input_hashes_after":inputs,
        "source_head":head,"dirty_scope":dirty,"argv":sys.argv,"python":sys.version,"platform":platform.platform(),
        "start_utc":start,"end_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifacts":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in [public,oracle]},
        "source_manufacture":"behavior-preserving C24 dyadic algorithm; fresh C25 IDs and seeds; UNIT includes labeled historical regression",
        "unit_fixture":args.split=="UNIT","regression_only_ids":[s["id"] for s in specs if s["id"].startswith("c24_")],"latent_oracles_passed_to_controller":False}
    with manifest.open("x") as out:out.write(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result))
if __name__=="__main__":main()
