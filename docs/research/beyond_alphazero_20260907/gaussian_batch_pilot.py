"""Exploratory two-finalist allocation lab; not an AlphaZero benchmark.

Run from any cwd with --output pointing to a NEW directory. All observations
have unit logical cost. Batch latency and neural/tree bias are not simulated.
Allocation depends only on covariance in this exact two-finalist Gaussian model.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import scipy
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
D = np.array([1., -1.])
NOISE = np.array([.04, .25])
MEAN = np.array([.1, 0.])
BUDGETS = (8, 32, 128, 512, 2048)
WIDTHS = (1, 8, 32)
METHODS = ('equal', 'neyman', 'stale_batch', 'pending_greedy')
SCENARIOS = (('independent', 0., 0.), ('positive', .9, .9),
             ('negative', -.6, -.6), ('misspecified', -.6, .9))


def gain(delta: float, shift_sd: float) -> float:
    """Exact E[max posterior means] increase for two actions, Gaussian shift."""
    if shift_sd <= 0:
        return 0.
    z = abs(delta) / shift_sd
    return max(0., shift_sd * math.exp(-.5*z*z)/math.sqrt(2*math.pi)
               - abs(delta)*float(ndtr(-z)))


def observe_cov(cov: np.ndarray, arm: int, noise: float) -> np.ndarray:
    col = cov[:, arm].copy()
    result = cov - np.outer(col, col)/(cov[arm, arm] + noise)
    return (result + result.T)/2


def shift_variance(cov: np.ndarray, arm: int, noise: float) -> float:
    return float((D @ cov[:, arm])**2/(cov[arm, arm] + noise))


def allocate(cov: np.ndarray, budget: int, width: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Two-action exact KG ordering equals one-step contrast variance ordering.

    pending_greedy conditions covariance on pending observations, never invents
    their values. stale_batch commits one whole batch with stale covariance.
    neyman is a strong known-noise baseline (asymptotically optimal allocation).
    """
    if budget < 1 or width < 1 or method not in METHODS:
        raise ValueError('positive budget/width and known method required')
    counts = np.zeros(2, dtype=int)
    post = cov.copy()
    ratio = np.sqrt(NOISE)/np.sqrt(NOISE).sum()
    for begin in range(0, budget, width):
        n = min(width, budget-begin)
        stale_arm = int(np.argmax([shift_variance(post, a, NOISE[a]) for a in range(2)]))
        for _ in range(n):
            if method == 'equal':
                arm = int(np.argmin(counts))
            elif method == 'neyman':
                arm = int(np.argmax((counts.sum()+1)*ratio-counts))
            elif method == 'stale_batch':
                arm = stale_arm
            else:
                arm = int(np.argmax([shift_variance(post, a, NOISE[a]) for a in range(2)]))
            counts[arm] += 1
            post = observe_cov(post, arm, NOISE[arm])
    return counts, post


def posterior(cov: np.ndarray, counts: np.ndarray, sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision = np.linalg.inv(cov)
    post = np.linalg.inv(precision + np.diag(counts/NOISE))
    means = (sums/NOISE + precision @ MEAN) @ post
    return means, post


def exact_bayes_regret(cov: np.ndarray, post: np.ndarray) -> float:
    prior_v = float(D @ cov @ D)
    explained = max(0., float(D @ (cov-post) @ D))
    return gain(float(D @ MEAN), math.sqrt(prior_v))-gain(float(D @ MEAN), math.sqrt(explained))


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def run(output: Path, worlds: int = 4096) -> None:
    start = time.perf_counter()
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('tracked source must be committed before execution')
    if output.exists():
        raise FileExistsError(f'refusing overwrite: {output}')
    output.mkdir(parents=True)
    summary = {'mode': 'exploratory_gaussian_two_finalist_diagnostic',
               'source_commit': git('rev-parse', 'HEAD'), 'source_tree': git('rev-parse','HEAD^{tree}'),
               'command': sys.argv, 'worlds_per_scenario': worlds,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'cpu': platform.machine(), 'noise_variances': NOISE.tolist(),
               'prior_mean': MEAN.tolist(), 'budgets': BUDGETS, 'batch_widths': WIDTHS,
               'methods': METHODS, 'rows': [], 'paired_comparisons': [],
               'raw_format': 'raw.csv.gz stores each unique (scenario,n0,n1) once; designs reference raw_key; floating values use 9 significant decimal digits'}
    # Gzip is deterministic (fixed mtime). Dedupe is statistical identity, not
    # deletion: all equal sufficient-statistic designs share the same raw_key.
    with (output/'raw.csv.gz').open('wb') as raw_file:
      with gzip.GzipFile(filename='', mode='wb', fileobj=raw_file, mtime=0) as compressed:
       import io
       with io.TextIOWrapper(compressed, encoding='utf8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['raw_key','unit','true_gap','estimated_gap','regret','error','covered95'])
        for scenario_i, (scenario, true_rho, assumed_rho) in enumerate(SCENARIOS):
            rng = np.random.default_rng(20260907+scenario_i)
            true_cov = .16*np.array([[1.,true_rho],[true_rho,1.]])
            cov = .16*np.array([[1.,assumed_rho],[assumed_rho,1.]])
            truth = rng.multivariate_normal(MEAN,true_cov,size=worlds)
            # Indexed common random numbers: each arm consumes its own prefix.
            noise_cumsum = rng.standard_normal((2,worlds,max(BUDGETS)))
            noise_cumsum *= np.sqrt(NOISE)[:,None,None]
            np.cumsum(noise_cumsum,axis=2,out=noise_cumsum)
            true_gap = truth @ D
            cache = {}
            losses = {}
            for width in WIDTHS:
             for budget in BUDGETS:
              for method in METHODS:
                counts, incremental_post = allocate(cov,budget,width,method)
                key = (int(counts[0]),int(counts[1]))
                raw_key = f'{scenario}:{key[0]}:{key[1]}'
                if key not in cache:
                    sums = truth*counts
                    for a in range(2):
                        if counts[a]:
                            sums[:,a] += noise_cumsum[a,:,counts[a]-1]
                    means, post = posterior(cov,counts,sums)
                    np.testing.assert_allclose(post,incremental_post,rtol=1e-10,atol=1e-12)
                    estimated_gap = means @ D
                    errors = (estimated_gap >= 0) != (true_gap >= 0)
                    regret = abs(true_gap)*errors
                    post_v = float(D @ post @ D)
                    covered = abs(estimated_gap-true_gap) <= 1.959963984540054*math.sqrt(post_v)
                    exact = exact_bayes_regret(cov,post) if true_rho == assumed_rho else None
                    cache[key] = (regret,errors,covered,post_v,exact)
                    for unit in range(worlds):
                        writer.writerow([raw_key,unit,format(true_gap[unit],'.9g'),format(estimated_gap[unit],'.9g'),
                                         format(regret[unit],'.9g'),int(errors[unit]),int(covered[unit])])
                regret,errors,covered,post_v,exact = cache[key]
                losses[(width,budget,method)] = regret
                summary['rows'].append({'scenario':scenario,'true_rho':true_rho,'assumed_rho':assumed_rho,
                    'width':width,'budget':budget,'method':method,'counts':counts.tolist(),'raw_key':raw_key,
                    'logical_evaluations':int(counts.sum()),'posterior_gap_variance':post_v,
                    'mean_regret':float(regret.mean()),'wrong_action_rate':float(errors.mean()),
                    'coverage95':float(covered.mean()),'exact_bayes_regret':exact})
              for baseline in METHODS[:-1]:
                diff = losses[(width,budget,'pending_greedy')]-losses[(width,budget,baseline)]
                se = float(diff.std(ddof=1)/math.sqrt(worlds))
                summary['paired_comparisons'].append({'scenario':scenario,'width':width,'budget':budget,
                   'contrast':f'pending_greedy minus {baseline}','mean_regret_difference':float(diff.mean()),
                   'descriptive_95_interval':[float(diff.mean()-1.96*se),float(diff.mean()+1.96*se)]})
            del noise_cumsum
    summary['elapsed_seconds'] = time.perf_counter()-start
    summary['raw_sha256'] = hashlib.sha256((output/'raw.csv.gz').read_bytes()).hexdigest()
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':'completed','source_commit':summary['source_commit'],'design_rows':len(summary['rows']),
                      'elapsed_seconds':summary['elapsed_seconds'],'raw_bytes':(output/'raw.csv.gz').stat().st_size}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worlds',type=int,default=4096)
    args = p.parse_args()
    if args.worlds < 2:
        p.error('--worlds must be >= 2')
    run(args.output,args.worlds)
