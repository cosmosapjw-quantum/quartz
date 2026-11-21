#!/usr/bin/env python3
"""External pretrained Rapfi mix9svq best-action reference for sealed freestyle15 roots.

2025 release executable and neural/classical weights are kept OUTSIDE the Git
artifact. Piskvork rule0 results are finite-budget alpha-beta teacher actions,
not exact minimax, and must NEVER enter live QUARTZ MCTS root control.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,os,re,select,subprocess,time

EXPECTED_WEIGHT={0:'mix9svqfreestyle_bsmix.bin.lz4',
                 1:'mix9svqstandard_bs15.bin.lz4',
                 4:'mix9svqrenju_bs15_black.bin.lz4'}

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def protocol_for_history(history,budget_ms,rule):
 if rule not in (0,1,4) or not 100<=budget_ms<=20000:
  raise ValueError('unfrozen rule or invalid teacher budget')
 if not isinstance(history,list) or len(history)>=225 or len(set(history))!=len(history) or any(type(a)!=int or a<0 or a>=225 for a in history):
  raise ValueError('bad source sequential 15x15 history')
 lines=['START 15',f'INFO rule {rule}',f'INFO timeout_turn {budget_ms}','BOARD']
 for i,a in enumerate(history):
  lines.append(f'{a%15},{a//15},{1 if i%2==0 else 2}')
 return '\n'.join(lines+['DONE'])+'\n'

def parse_move_and_model(stdout,rule):
 if rule not in EXPECTED_WEIGHT:raise ValueError('invalid rule variant')
 errors=[s for s in stdout.splitlines() if s.startswith('ERROR')]
 if errors:raise ValueError('Rapfi signalled error: '+';'.join(errors))
 match=re.findall(r'(?m)^([0-9]{1,2}),([0-9]{1,2})\s*$',stdout)
 if not match:raise ValueError('no completed Piskvork move')
 x,y=map(int,match[-1])
 if not (0<=x<15 and 0<=y<15):raise ValueError('off-board reference')
 expected=EXPECTED_WEIGHT[rule]
 return {
  'xy':[x,y],'flat_action':y*15+x,
  'compatible_weight_reported':f'load weight from {expected}' in stdout,
  'saw_loaded_nnue':('mix9svq nnue: load weight from' in stdout),
  'loaded_expected_weight_name':expected,
  'engine_error_messages':[],
 }

def query_one(exe,root,history,rule,budget_ms,hard_deadline_s=15):
 wire=protocol_for_history(history,budget_ms,rule)
 proc=subprocess.Popen([str(exe)],cwd=root,stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,start_new_session=True,
    env={**os.environ,'OMP_NUM_THREADS':'1'})
 began=time.perf_counter()
 output=b''
 status='TIMEOUT'
 try:
  proc.stdin.write(wire.encode());proc.stdin.flush()
  while time.perf_counter()-began<hard_deadline_s:
   if proc.poll() is not None:break
   ready,_,_=select.select([proc.stdout],[],[],.6)
   if not ready:continue
   chunk=os.read(proc.stdout.fileno(),65536)
   if not chunk:break
   output+=chunk
   if len(output)>250000:
    status='TOO_MUCH_OUTPUT';break
   if re.search(rb'(^|\n)[0-9]{1,2},[0-9]{1,2}\r?\n',output):
    status='OK';break
  stdout=output.decode('utf-8',errors='replace')
  elapsed_ms=(time.perf_counter()-began)*1000
  if status=='OK':
   try:
    parsed=parse_move_and_model(stdout,rule)
    if parsed['flat_action'] in history:
     raise ValueError('Rapfi selected an already occupied cell')
    if not parsed['compatible_weight_reported']:
     raise ValueError('loaded NNUE weight identity missing from output')
   except ValueError as exc:
    return {'status':'INVALID_ENGINE_OUTPUT','error':str(exc),
            'elapsed_wall_ms':elapsed_ms,'stdout_prefix':stdout[:3500]}
   return {'status':'OK','elapsed_wall_ms':elapsed_ms,
           'wall_budget_requested_ms':budget_ms,
           'stdout_prefix':stdout[:3500],**parsed}
  return {'status':status if status!='TIMEOUT' or proc.poll() is None else 'EXIT_BEFORE_MOVE',
          'elapsed_wall_ms':elapsed_ms,'stdout_prefix':stdout[:3500],
          'exitcode_before_end':proc.poll()}
 finally:
  try:proc.stdin.write(b'END\n');proc.stdin.flush()
  except (BrokenPipeError,ValueError):pass
  try:proc.wait(timeout=2)
  except subprocess.TimeoutExpired:
   proc.kill();proc.wait()

def main():
 cli=argparse.ArgumentParser(description=__doc__)
 cli.add_argument('--out-dir',type=Path,required=True)
 args=cli.parse_args()
 d=Path(__file__).resolve().parent.parent
 repo=next(p for p in d.parents if (p/'.git').exists())
 if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
  raise RuntimeError('Rapfi teacher only runs on clean, previously published source')
 locked=json.loads((d/'coding/C01_INPUT_LOCK.json').read_text())
 rec={r['name']:r for r in locked['sources']}
 refs=('Rapfi_executable','Rapfi_config','Rapfi_classical_model','Rapfi_NNUE_freestyle')
 for name in refs:
  p=Path(rec[name]['path'])
  if not p.is_file() or p.stat().st_size!=rec[name]['bytes'] or sha(p)!=rec[name]['sha256']:
   raise RuntimeError('Rapfi source or pretrained weights changed: '+name)
 exe=Path(rec['Rapfi_executable']['path'])
 original=exe.parent
 cohortfile=d/'COHORT_24_FROZEN.json'
 cohort=json.loads(cohortfile.read_text())
 assert len(cohort['rows'])==24
 assert locked['teacher_timeout_turn_ms']==300
 assert locked['teacher_secondary_budget_ms']==600
 all_rows=[]
 for source in cohort['rows']:
  first=query_one(exe,original,source['history'],0,locked['teacher_timeout_turn_ms'])
  second=query_one(exe,original,source['history'],0,locked['teacher_secondary_budget_ms'])
  all_rows.append({'id':source['id'],'stratum':source['stratum'],
                   'history':source['history'],'rule':0,
                   'reference_300ms':first,'stability_600ms':second,
                   'both_budgets_same_action':first.get('status')=='OK' and second.get('status')=='OK' and first['flat_action']==second['flat_action']})
  print('RAPFI_REAL_TRAINED_REFERENCE',source['id'],first.get('status'),
        first.get('flat_action'),'secondary',second.get('status'),second.get('flat_action'),flush=True)
 for name in refs:
  if sha(Path(rec[name]['path']))!=rec[name]['sha256']:
   raise RuntimeError('source pretrained model bytes mutated during reference run')
 status_primary=Counter(x['reference_300ms']['status'] for x in all_rows)
 status_second=Counter(x['stability_600ms']['status'] for x in all_rows)
 result={
  'schema':'quartz_cog04_highbranch14_public_trained_Rapfi_reference_v1',
  'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
  'source_strata_24_sha256':sha(cohortfile),
  'rule':'FREESTYLE_15','Piskvork_rule_number':0,
  'Rapfi_binary_sha256':rec['Rapfi_executable']['sha256'],
  'Rapfi_config_sha256':rec['Rapfi_config']['sha256'],
  'Rapfi_classical_model_sha256':rec['Rapfi_classical_model']['sha256'],
  'Rapfi_trained_NNUE_sha256':rec['Rapfi_NNUE_freestyle']['sha256'],
  'release_origin':'local official Rapfi 250615 packaged engine+matching model',
  'teacher_is_finite_depth_alpha_beta_not_exact_game_WDL':True,
  'no_Rapfi_labels_passed_to_QUARTZ_at_evaluation_time':True,
  'reference_300ms_status_counts':dict(status_primary),
  'stability_600ms_status_counts':dict(status_second),
  'same_action_across_two_budgets_count':sum(x['both_budgets_same_action'] for x in all_rows),
  'rows':all_rows
 }
 out=args.out_dir.resolve()
 out.mkdir(parents=True,exist_ok=False)
 data=out/'RAPFI_REFERENCE_24.json'
 data.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
 (out/'EXECUTION_MANIFEST.json').write_text(json.dumps({
  'schema':'quartz_cog04_C07_Rapfi_reference_manifest_v1',
  'source_commit':result['source_commit'],
  'frozen_C14_source_sha256':result['source_strata_24_sha256'],
  'Rapfi_model_hash':result['Rapfi_trained_NNUE_sha256'],
  'Rapfi_binary_hash':result['Rapfi_binary_sha256'],
  'result_file_sha256':sha(data),
  'status_primary':dict(status_primary),
  'status_secondary':dict(status_second),
  'n_reference_roots':len(all_rows),'not_an_exact_oracle':True,
 },indent=2,sort_keys=True)+'\n')
 print('C14_C07_ACTUAL_24_TRULY_PRETRAINED_RAPFI_REFERENCES',
       json.dumps({'primary':dict(status_primary),'secondary':dict(status_second),
         'budget_agreement':result['same_action_across_two_budgets_count']},sort_keys=True))

if __name__=='__main__':main()
