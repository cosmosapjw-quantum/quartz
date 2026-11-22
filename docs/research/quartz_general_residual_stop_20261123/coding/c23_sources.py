"""Exact dyadic manufactured SPD sources for Quartz C23 (no solver outcomes used).

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
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['TRAIN','CAL','TEST'],required=True);args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1]
    config=json.loads((doc/'SOURCE_SPECS.json').read_text())
    specs=[r for r in config['sources'] if r['split']==args.split]
    public=doc/'evidence'/('C23_PUBLIC_'+args.split+'.jsonl.gz')
    oracle=doc/'evidence'/('C23_ORACLE_'+args.split+'.jsonl.gz')
    manifest=doc/'evidence'/('C23_SOURCES_'+args.split+'.json')
    if any(p.exists() for p in [public,oracle,manifest]):raise FileExistsError('source output exists; preserve and inspect')
    rows=[make_source(s) for s in specs]
    write_gzip(public,[r[0] for r in rows]);write_gzip(oracle,[r[1] for r in rows])
    result={'split':args.split,'count':len(rows),'source_specs_sha256':hashlib.sha256((doc/'SOURCE_SPECS.json').read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [public,oracle]},'latent_oracles_passed_to_controller':False}
    manifest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
