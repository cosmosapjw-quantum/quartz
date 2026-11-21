#!/usr/bin/env python3
"""C5/C6 tests independent of teacher labels or heldout adaptation."""
import unittest
import numpy as np
from policy15 import one_feature,position_features,stable_prob,nll_grad_hessian,FEATURE_ORDER
from freeze_cohorts import replay,win,canon
class TestFinitePolicy13(unittest.TestCase):
 def test_feature_contract_and_tactical_witness(self):
  board=[0]*225
  f=one_feature(board,1,0)
  self.assertEqual(len(f),len(FEATURE_ORDER))
  self.assertEqual(f[0],1)
  self.assertEqual(f[1],0)
  self.assertEqual(f[2],0)
  self.assertEqual(f[7],0.2)
  self.assertEqual(f[8],0.2)
  self.assertFalse(f[11])
  for a in [31,32,33,34]:board[a]=1
  own=one_feature(board,1,30)
  opponent=one_feature(board,2,30)
  self.assertEqual(own[11],1)
  self.assertEqual(opponent[12],1)
  self.assertEqual(own[7],1)
  self.assertEqual(own[9],1)
 def test_legal_softmax_and_grad_hess(self):
  rng=np.random.default_rng(15)
  X=rng.random((25,13))
  t=rng.uniform(-.5,.5,13)
  y=11
  p=stable_prob(X@t)
  self.assertTrue(np.isclose(p.sum(),1))
  self.assertTrue(np.all(p>0))
  loss,g,h=nll_grad_hessian(t,[X],[y],.01)
  eye=np.eye(13)
  numeric=np.array([(nll_grad_hessian(t+1e-6*eye[i],[X],[y],.01)[0]-
                     nll_grad_hessian(t-1e-6*eye[i],[X],[y],.01)[0])/2e-6 for i in range(13)])
  self.assertLess(float(np.max(np.abs(g-numeric))),1e-8)
  self.assertGreater(float(np.linalg.eigvalsh(h).min()),0.00999)
 def test_source_legality_and_rejection(self):
  with self.assertRaises(ValueError):
   replay([31,31])
  with self.assertRaises(ValueError):
   one_feature([0]*225,1,225)
  b=replay([31,160,32,177,33,190,34,207])
  self.assertTrue(win(b,1,30))
  acts,feats=position_features([31,160])
  self.assertEqual(len(acts),223)
  self.assertEqual(feats.shape,(223,13))
  self.assertFalse(31 in acts)
 def test_features_are_D4_equivariant_by_action(self):
  b=[0]*225
  for a in [31,33,62,77,80,154]:b[a]=1
  for a in [95,191,136,34,17]:b[a]=2
  a=37
  f=one_feature(b,1,a)
  for mirror in (False,True):
   for rot in range(4):
    def transformed(x,y):
     if mirror:x=14-x
     for _ in range(rot):x,y=14-y,x
     return x,y
    board=[0]*225
    for j,s in enumerate(b):
     x,y=transformed(j%15,j//15)
     board[y*15+x]=s
    x,y=transformed(a%15,a//15)
    np.testing.assert_allclose(f,one_feature(board,1,y*15+x),atol=0,rtol=0)
if __name__=="__main__": unittest.main()
