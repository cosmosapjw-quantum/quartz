#!/usr/bin/env python3
"""C29 scoped native protocol and independent exact-world regression tests."""
import hashlib,importlib.util,json,os,random,subprocess,sys,unittest
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/research/quartz_c29_noisy_query_20261123'
BIN=Path(os.environ.get('C29_BINARY','/home/cosmosapjw/quantum/research/quartz_c29_target_20261123/release/quartz_c29_noisy_query'))
spec=importlib.util.spec_from_file_location('quartz_c29_exact_oracle',DOC/'theory/noisy_exact.py')
oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)

def native(specification,h=2,id='test'):
    return dict(mode='synthetic',id=id,
        probabilities=[[float(q) for q in p] for p in specification['p']],
        loadings=[float(q) for q in specification['r']],
        costs=[float(q) for q in specification['c']],
        channels=[[[float(q) for q in row] for row in channel] for channel in specification['ch']],
        m=float(specification['m']),b=float(specification['b']),
        horizon=h,known_only=specification['known_only'])

def run_lines(lines):
    assert BIN.is_file()
    p=subprocess.run([str(BIN)],input='\n'.join(lines)+'\n',
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     text=True,timeout=180)
    assert p.returncode==0,(p.returncode,p.stderr)
    assert p.stderr=='',p.stderr
    output=p.stdout.splitlines()
    assert len(output)==len(lines),(len(output),len(lines))
    return [json.loads(v) for v in output]

def identity():
    return tuple(tuple(F(int(i==y)) for y in range(3)) for i in range(3))

def bsc():
    return ((F(3,4),F(1,4)),(F(1,2),F(1,2)),(F(1,4),F(3,4)))

class NativeTests(unittest.TestCase):
    def test_canonical_fixtures(self):
        one=oracle.model([(F(1,2),F(0),F(1,2))],[0],[bsc()],[F(1,100)],b=F(3,5))
        two=oracle.model([(F(1,2),F(0),F(1,2))]*2,[0,0],[bsc()]*2,
                         [F(1,20)]*2,b=0)
        exact=oracle.model([(F(1,3),)*3]*2,[F(1,2),F(-1,2)],[identity()]*2,
                           [F(1,100)]*2,b=F(1,2),known_only=True)
        requests=[native(one,1,'oneH1'),native(one,2,'oneH2'),native(two,1,'twoH1'),
                  native(exact,2,'compat')]
        responses=run_lines([json.dumps(x) for x in requests])
        for a,s,h in zip(responses,[one,one,two,exact],[1,2,1,2]):
            expected,choice=oracle.compare(s,h)
            self.assertEqual(a['choice'],choice)
            self.assertLessEqual(abs(a['value']-float(expected)),5e-11)
        self.assertAlmostEqual(responses[1]['value']-responses[0]['value'],11/800,places=12)

    def test_random_exact_world_100(self):
        rng=random.Random(2963)
        requests=[];expected=[];uncertain=0
        for k in range(100):
            n=rng.randrange(1,5)
            ps=[]
            for i in range(n):
                weights=[rng.randint(0,5) for _ in range(3)]
                if sum(weights)==0:weights[0]=1
                ps.append(tuple(F(x,sum(weights)) for x in weights))
            chan=[]
            for i in range(n):
                cols=rng.choice([2,3])
                matrix=[]
                for u in range(3):
                    weights=[rng.randint(0,6) for _ in range(cols)]
                    if sum(weights)==0:weights[0]=1
                    matrix.append(tuple(F(v,sum(weights)) for v in weights))
                chan.append(tuple(matrix))
            rr=[rng.choice([F(-3,4),F(-1,2),F(0),F(1,2),F(3,4)]) for _ in range(n)]
            spec=oracle.model(ps,rr,chan,
                [rng.choice([F(0),F(1,100),F(1,25),F(1,5)]) for _ in range(n)],
                b=rng.choice([F(0),F(1,2),F(3,5),F(1)]),
                m=rng.choice([F(-1),F(-1,2),F(0),F(1,2),F(1)]))
            horizon=rng.choice([0,1,2])
            v,choice=oracle.compare(spec,horizon)
            requests.append(native(spec,horizon,f'random{k}'))
            expected.append((v,choice))
        responses=run_lines([json.dumps(x,allow_nan=False) for x in requests])
        for a,(v,choice) in zip(responses,expected):
            self.assertNotIn('error',a)
            self.assertLessEqual(abs(a['value']-float(v)),5e-11)
            if a['choice']!=choice: # possible epsilon tie, not an invalid Bellman value
                self.assertLessEqual(abs(a['value']-a['stop']),1e-12)
                uncertain+=1
        self.assertLessEqual(uncertain,3,'unexpected float action differences')

    def test_no_information_and_zero_support(self):
        noinfo=tuple((F(1,2),F(1,2)) for _ in range(3))
        samples=[]
        for p in [(F(1,2),F(0),F(1,2)),(F(0),F(0),F(1))]:
            s=oracle.model([p],[F(0)],[noinfo],[F(1,100)],b=F(3,5))
            samples.append(native(s,2))
        out=run_lines([json.dumps(x) for x in samples])
        self.assertEqual(out[0]['choice'],None)
        self.assertEqual(out[1]['choice'],None)
        self.assertAlmostEqual(out[0]['value'],0.6)
        self.assertAlmostEqual(out[1]['value'],1.0)

    def test_invalid_request_does_not_poison_stream(self):
        good=native(oracle.model([(F(1,2),F(0),F(1,2))],[0],[bsc()],[F(1,100)],b=F(3,5)),2)
        bad=[dict(good,probabilities=[[0.6,0.0,0.6]]),
             dict(good,channels=[[[0.7,0.7],[0.5,0.5],[0.25,0.75]]]),
             dict(good,loadings=[1.0]),
             dict(good,costs=[-0.1]),
             dict(good,horizon=3),
             dict(good,known_only=True),
             dict(good,unused_unknown=123)]
        reqs=[json.dumps(x) for x in bad]+['{invalid']+[json.dumps(good)]
        out=run_lines(reqs)
        self.assertTrue(all('error' in x for x in out[:-1]))
        self.assertNotIn('error',out[-1])
        self.assertAlmostEqual(out[-1]['value'],0.67875)

if __name__=='__main__':
    unittest.main()
