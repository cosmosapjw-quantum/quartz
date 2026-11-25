import json
from engine import Engine,D,identity
from rules import fixtures
from native_smoke import check
h=fixtures()['long']
with Engine(1) as e:
    a=e.search(h,'freestyle','BASE',8,1,17)
    check(a,h,'freestyle')
    assert a['python_cpu_ns']>0 and a['rust_cpu_ticks']>=0 and a['tick_hz']>0 and a['nn_rows']==9
    out=dict(identity=identity(),fixture='previous constructed150-ply state; no move played',row=a)
(D/'C08_CPU_PROBE.json').write_text(json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
print('CPU_PROBE',a['python_cpu_ns'],a['rust_cpu_ticks'],a['tick_hz'],a['transport_ns'],a['nn_rows'])
