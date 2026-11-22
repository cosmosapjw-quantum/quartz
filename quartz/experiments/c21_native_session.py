"""C21 domain-neutral native computation observation harness.

* beta_eval: strictly initial-only posterior / planned observation budget.
* bench_task: separate OFFLINE source simulator using latent true utility.
* Long-lived JSONL server includes Python JSON encoding, stdin/pipe crossing,
  Rust parsing/serialization and response parsing in measured host roundtrip.
  Native function-only costs are returned separately by Rust Instant.
"""
from pathlib import Path
from typing import Mapping,Sequence
import hashlib,json,math,subprocess,tempfile,time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
BETA=ROOT/"target/release/quartz_c21_fast_evsi"
SENSOR=Path("/home/cosmosapjw/quantum/research/quartz_general_meta20_20261122/target/release/quartz_c20_shadow")
SENSOR_SHA256="7943f2719d26793f0d73c0754933aba14693b970cb8f03a0f594acd3ba6e15f5"
def sha(path):
 with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def source_only_beta(row:Mapping)->dict:
 assert row["family"]=="beta_arms"
 observations=row["initial_observations"]
 assert len(observations)==row["n_actions"] and all(len(x)==3 for x in observations)
 posterior=[[float(2+sum(seq)),float(2+len(seq)-sum(seq))] for seq in observations]
 return {"posterior":posterior,"extra_per_arm":[10]*len(posterior)}
def offline_bench_beta(row:Mapping)->dict:
 case=source_only_beta(row)
 # This exact source theta is permitted only for OFFLINE stochastic task execution.
 true=row["theta_offline_true_arm_means"]
 assert len(true)==len(case["posterior"])
 text=f"C21-NATIVE-TASK:{row['group_id']}:{row['rep']}:{row['split']}"
 seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")
 return dict(case,theta_offline_means=true,seed=seed)

def sensor_public_case(row:Mapping)->dict:
 assert row["family"]=="finite_sensor"
 b=list(row["prior_PUBLIC"])
 L=[list(p) for p in row["likelihood_PUBLIC"]]
 U=[list(u) for u in row["utilities_PUBLIC"]]
 init=row["initial_observations"]
 for y in init:
  vals=[b[i]*L[i][y] for i in range(len(b))]
  z=math.fsum(vals)
  if z<=0:raise ValueError("impossible initial observed categorical sensor outcome")
  b=[v/z for v in vals]
 return {"prior":b,"likelihood":L,"utilities":U,"extra":8}

class NativeBetaSession:
 def __enter__(self):
  self.launch_ns=time.perf_counter_ns()
  self.proc=subprocess.Popen([str(BETA),"--server"],
     text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=1)
  warm={"cmd":"beta_eval","cases":[{"posterior":[[1.,1.],[1.,1.]],"extra_per_arm":[1,1]}]}
  result,self.startup_first_request_us=self.call(warm)
  assert result["schema"]=="quartz_C21_beta_exact_native_v1"
  return self
 def call(self,request:Mapping)->tuple[dict,float]:
  t0=time.perf_counter_ns()
  self.proc.stdin.write(json.dumps(request,sort_keys=True,separators=(",",":"))+"\n")
  self.proc.stdin.flush()
  answer=self.proc.stdout.readline()
  if not answer:raise RuntimeError("native source-only Rust C21 server exited prematurely")
  response=json.loads(answer)
  elapsed=max(1,time.perf_counter_ns()-t0)/1000
  if "error" in response:raise RuntimeError("native C21 source input rejected: "+response["error"])
  return response,elapsed
 def __exit__(self,typ,value,tb):
  try:
   self.proc.stdin.write('{"cmd":"quit"}\n');self.proc.stdin.flush()
  except (BrokenPipeError,OSError):pass
  self.proc.stdin.close()
  self.proc.wait(timeout=20)
  if self.proc.returncode!=0 and typ is None:
    raise RuntimeError("native C21 Rust server unexpectedly exited nonzero")

def native_beta_with_real_task(rows:Sequence[Mapping])->tuple[list[dict],dict]:
 out=[]
 with NativeBetaSession() as s:
  startup=s.startup_first_request_us
  for row in rows:
   case=source_only_beta(row)
   pred,eval_us=s.call({"cmd":"beta_eval","cases":[case]})
   assert pred["source_truth_used"] is False and pred["no_SearchPolicy_mutation"] is True
   val=pred["results"][0]
   task,task_server_us=s.call({"cmd":"bench_task","cases":[offline_bench_beta(row)]})
   assert task["source_truth_is_offline_sampler_only"] is True
   actual=task["results"][0]
   theta=row["theta_offline_true_arm_means"]
   a0=int(row["pre_action"]);a1=int(actual["sampled_terminal_action"])
   assert a0 in range(len(theta)) and a1 in range(len(theta))
   out.append({
    "source_parent":row["group_id"],"rep":row["rep"],"split":row["split"],
    "before_action":a0,"after_native_extra_action":a1,
    "actual_native_source_signed_gain":float(theta[a1]-theta[a0]),
    "actual_native_source_revision":int(a0!=a1),
    "actual_native_extra_task_internal_us":float(actual["task_function_us"]),
    "observed_task_bernoulli_draws":int(actual["task_num_actual_bernoulli_draws"]),
    "native_offline_task_server_roundtrip_us":task_server_us,
    "gross_exact_EVSI":float(val["gross_evsi"]),
    "before_posterior_expected_best":float(val["before_best"]),
    "exact_predictive_mass":float(val["joint_mass"]),
    "unique_support_points":int(val["support_points"]),
    "events":int(val["events"]),
    "native_predictor_function_internal_us":float(val["function_internal_us"]),
    "native_predictor_streaming_roundtrip_us":float(eval_us),
    "oracle_data_in_forecast_request":False,
    "source_task_sampler_secret_offline_only":True,
   })
 return out,{"server_first_request_cold_start_us":startup,"server_mode":"long_lived_one_case_per_request",
    "n_test_episodes":len(rows)}

def native_sensor_model_batch(rows:Sequence[Mapping])->tuple[list[dict],dict]:
 if sha(SENSOR)!=SENSOR_SHA256:raise ValueError('immutable C20 native sensor source SHA mismatch')
 cases=[sensor_public_case(row) for row in rows]
 assert all(set(r)=={"prior","likelihood","utilities","extra"} for r in cases)
 with tempfile.TemporaryDirectory(prefix="quartz_c21_sensor_input_") as tmp:
  path=Path(tmp)/"C21_SENSOR_MODEL_ONLY.json"
  path.write_text(json.dumps({"cases":cases},sort_keys=True))
  t0=time.perf_counter_ns()
  raw=subprocess.check_output([str(SENSOR),str(path)],text=True)
  elapsed=(time.perf_counter_ns()-t0)/1000
 record=json.loads(raw)
 assert record["no_SearchPolicy_mutation"] is True and record["beta_binomial_Rust_port"] is False
 assert len(record["predictions"])==len(rows)
 out=[{"gross_exact_EVSI":float(v["evsi"]),
       "native_predictor_function_internal_us":float(v["evaluation_us"]),
       "before_posterior_expected_best":float(v["before_best"]),
       "exact_predictive_mass":float(v["total_predictive_mass"]),
       "unique_support_points":int(v["outcome_count"])}
      for v in record["predictions"]]
 return out,{"total_cold_batch_host_us":elapsed,
   "amortized_batch_host_us_per_case":elapsed/len(rows),
   "sensor_model_host_scope":"one Rust process for all cases; no per-case streaming/JSON boundary"}
