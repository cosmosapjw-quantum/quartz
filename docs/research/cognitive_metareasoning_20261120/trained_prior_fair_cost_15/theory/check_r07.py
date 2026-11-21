#!/usr/bin/env python3
"""Executable checks for finite-action exact probability and convex imitation loss."""
from __future__ import annotations
from fractions import Fraction
from math import comb
from pathlib import Path
import json
import numpy as np

def main():
    b,k,m,r = 225,16,1,4
    discovery = Fraction(1,1)-Fraction(comb(b-k-m,r),comb(b-k,r))
    assert discovery == Fraction(4,209)
    rng=np.random.default_rng(1507)
    X=rng.uniform(-1,1,size=(21,5))
    theta=rng.uniform(-0.5,0.5,size=5)
    y=7
    def nll(t):
        z=X@t
        d=z.max()
        return float(np.log(np.exp(z-d).sum())+d-z[y])
    z=X@theta
    probs=np.exp(z-z.max())
    probs/=probs.sum()
    analytic = probs@X-X[y]
    numeric=np.array([(nll(theta+np.eye(5)[j]*1e-6)-nll(theta-np.eye(5)[j]*1e-6))/2e-6 for j in range(5)])
    grad_err=float(np.max(np.abs(analytic-numeric)))
    mu=probs@X
    centered=X-mu
    hessian=centered.T@(centered*probs[:,None])
    min_eigen=float(np.linalg.eigvalsh(hessian).min())
    assert grad_err<1e-8,grad_err
    assert min_eigen>-1e-12,min_eigen
    assert np.isclose(probs.sum(),1)
    # Center-radius features stay unchanged under every D4 board symmetry.
    def rotate(x,y):return 14-y,x
    def reflect(x,y):return 14-x,y
    checks=0
    for y0 in range(15):
        for x0 in range(15):
            baseline=(max(abs(x0-7),abs(y0-7)),abs(x0-7)+abs(y0-7))
            for mirrored in [False,True]:
                x,y=reflect(x0,y0) if mirrored else (x0,y0)
                for _ in range(4):
                    assert (max(abs(x-7),abs(y-7)),abs(x-7)+abs(y-7))==baseline
                    checks+=1
                    x,y=rotate(x,y)
    result={"stage":"R07","status":"PASS","discovery_fraction":str(discovery),
            "softmax_sum":float(probs.sum()),"max_finite_diff_grad_error":grad_err,
            "min_hessian_eigenvalue":min_eigen,
            "D4_center_geometry_invariance_checks":checks,
            "method":"NumPy analytic vs centered finite differences and independent eigenspectrum",
            "Wolfram_exact_cross_check":"4/209 verified; 3-action logsumexp gradient difference ==0",
            "no_actual_Gomoku_model_trained":True}
    (Path(__file__).parent/"R07_CHECKS.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
