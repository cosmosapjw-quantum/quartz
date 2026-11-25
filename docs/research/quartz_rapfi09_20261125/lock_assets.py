import json,hashlib,subprocess
from pathlib import Path
from publish import D,ROOT
p=ROOT/'docs/research/cognitive_metareasoning_20261120/highbranch_transfer_14/coding/C01_INPUT_LOCK.json'
prior=json.loads(p.read_text())
names={'Rapfi_executable','Rapfi_config','Rapfi_classical_model','Rapfi_NNUE_freestyle'}
rows=[x for x in prior['sources'] if x['name'] in names]
assert len(rows)==4
for r in rows:
    p=Path(r['path'])
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
    assert p.stat().st_size==r['bytes']
r=ROOT.parent/'quartz_external_engines_20261120/rapfi'
tag=subprocess.check_output(['git','-C',str(r),'rev-parse','refs/tags/250615'],text=True).strip()
assert tag=='1be1551ced57e38d53ed58f6d74bf6f8b4bdc230'
(D/'RAPFI_LOCK.json').write_text(json.dumps(dict(tag='250615',source_commit=tag,source_url='https://github.com/dhbloo/rapfi',release_url='https://github.com/dhbloo/rapfi/releases/tag/250615',license='GPL-3.0',files=rows),indent=2,sort_keys=True)+'\n')
print('Verified four original external asset identities and exact source tag',tag)
