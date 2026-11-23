"""Build the final noncircular publication snapshot and artifact manifest."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/research/quartz_general_incumbent_price_20261123'
BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta25_backups')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
base=json.loads((DOC/'BASELINE.json').read_text())
receipts=[json.loads(x) for x in (BACKUP/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
assert len(receipts)==20 and receipts[-1]['stage']=='C09'
assert git('rev-parse','HEAD')==receipts[-1]['commit']=='61707611fdf3319bb5c0d7f26da7ff8cf6ac5254'
freeze=json.loads((DOC/'evidence/C07_PRE_TEST_FREEZE.json').read_text())
for p,want in freeze['frozen_inputs'].items():assert sha(ROOT/p)==want,p
claims=json.loads((DOC/'CLAIMS.json').read_text())
for claim in claims['claims']:
 ev=claim.setdefault('evidence',[])
 if claim['id'] in ['T1','T2','T3','T4']:
  for p in ['review/C09_REVIEW.md','review/C09_REVIEW_CHECKS.json']:
   if p not in ev:ev.append(p)
 if claim['id']=='T5':
  for p in ['ORIGIN_WATCHDOG_RESULT.json','NEXT_STAGE.json']:
   if p not in ev:ev.append(p)
claims['closure']={'scientific_snapshot':freeze.get('source_head','21c71da38ba1243d92f4f7682f54ef485eb5c3c6'),'outcome_snapshot':'d45173685f955820bd848bf4ddb2044382ca7e67','review_gate':'PASS_FOR_DECLARED_SCOPE','primary_economic_benefit':'NOT_SUPPORTED_ON_FRESH_TEST','C26_executed':False,'final_publication_identity':'external C10 receipt and FINAL_PUBLICATION_AUDIT.json'}
(DOC/'CLAIMS.json').write_text(json.dumps(claims,indent=2)+'\n')
snapshot={'schema':'quartz.c25.stage-publication-snapshot.v1','through_stage':'C09','through_commit':receipts[-1]['commit'],'receipts':receipts,'final_C10_receipt':'External backup/STAGE_RECEIPTS.jsonl; appended after the commit and verified by FINAL_PUBLICATION_AUDIT.json. Excluded here to avoid self-reference.','client_date':'2026-11-23 Asia/Seoul'}
p=DOC/'STAGE_PUBLICATIONS.json';assert not p.exists();p.write_text(json.dumps(snapshot,indent=2)+'\n')
mp=DOC/'ARTIFACTS.json';assert not mp.exists()
missing=[]
for name in ['REPORT.md','REPRODUCE.md','README.md']:
 for dest in re.findall(r'\]\(([^)]+)\)',(DOC/name).read_text()):
  if not dest.startswith(('https:','http:','#')) and dest!='ARTIFACTS.json' and not (DOC/dest.split('#')[0]).exists():missing.append([name,dest])
assert not missing,missing
for name in ['NEXT_STAGE.json','ORIGIN_WATCHDOG_RESULT.json','CLAIMS.json']:
 json.loads((DOC/name).read_text())
subprocess.run(['git','add','--',str(DOC.relative_to(ROOT))],cwd=ROOT,check=True)
changed=git('diff','--name-only',base['parent']).splitlines();files={}
for rel in changed:
 assert rel.startswith('docs/research/quartz_general_incumbent_price_20261123/') or rel in ['src/bin/quartz_c25_incumbent_price.rs','tests/test_c25_sources.py','scripts/quartz_general_c25_publish.py'],rel
 p=ROOT/rel;assert p.is_file();files[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
manifest={'schema':'quartz.c25.artifacts.v1','relative_to':'repository root','parent':base['parent'],'self_excluded':str(mp.relative_to(ROOT)),'file_count':len(files),'bytes':sum(v['bytes'] for v in files.values()),'files':files}
mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'C10_PACKAGE_PREPARED','pre_C10_receipts':len(receipts),'manifest_files_excluding_self':len(files),'manifest_bytes_excluding_self':manifest['bytes'],'frozen_inputs_unchanged':len(freeze['frozen_inputs']),'document_links_checked':True,'C26_executed':False}))
