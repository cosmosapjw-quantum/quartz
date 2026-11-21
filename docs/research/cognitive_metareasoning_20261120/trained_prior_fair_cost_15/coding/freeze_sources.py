#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,platform
d=Path(__file__).resolve().parent
root=next(x for x in d.parents if (x/'.git').exists())
c14=root/'docs/research/cognitive_metareasoning_20261120/highbranch_transfer_14/coding/C01_INPUT_LOCK.json'
old=json.loads(c14.read_text())
sources=[]
for x in old['sources']:
  if x['name'].startswith('Rapfi'):
    path=Path(x['path'])
    assert path.is_file(),x
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    assert h==x['sha256'] and path.stat().st_size==x['bytes'], x
    sources.append({'name':x['name'],'path':str(path),'bytes':path.stat().st_size,'sha256':h,'distribution':'external_read_only'})
for rel in ['src/games/gomoku15.rs','src/mcts_server/highbranch14_cpu_server.rs','src/mcts/select.rs','src/gomocup_bundle.rs']:
  p=root/rel
  sources.append({'name':rel,'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'distribution':'repository'})
freeze=root/'docs/research/cognitive_metareasoning_20261120/highbranch_transfer_14/COHORT_24_FROZEN.json'
sources.append({'name':'C14_heldout_exclusion','path':str(freeze),'bytes':freeze.stat().st_size,'sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'distribution':'do_not_train'})
result={'schema':'COG05_C01_immutable_source_preflight_v1',
'parent_git_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
'game':'Freestyle Gomoku 15x15; no gravity; >=5',
'teacher_role':'finite-time external Rapfi move-label source; never online MCTS or exact oracle',
'model':'new policy-only 13-feature log-linear learner; no existing trained compatible ONNX',
'train_positions':128,'validation_positions':32,'heldout_natural':48,'heldout_tactical_diagnostic':16,
'teacher_primary_budget_ms':300,'teacher_secondary_budget_ms':600,
'CPU_target_ms':20,'CPU_mean_mismatch_gate_per_stratum':0.10,
'comparators':['learned_gated','learned_all_legal','uniform_all_legal'],
'source':sources,'python':platform.python_version(),
'git_backup':'every stage pushed verified and archived locally',
'external_Rapfi_model_redistribution':False}
(d/'C01_INPUT_LOCK.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('C01_LOCK',json.dumps({'sources':len(sources),'source_branch_parent':result['parent_git_commit']}))
