"""Train/eval contracts for spatial model and strictly heldout family splits."""
from pathlib import Path
import math
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generalize_cpu import ConvPolicyValue, model_to_portable, compare_portable, legal_masked_scores, validate_splits

class Generalization07ModelTests(unittest.TestCase):
    def test_conv_model_has_shape_and_export_parity(self):
        torch.manual_seed(17)
        net=ConvPolicyValue().eval()
        inputs=torch.rand((2,126),dtype=torch.float32)
        with torch.inference_mode():
            logits,value=net(inputs)
        self.assertEqual(tuple(logits.shape),(2,7))
        self.assertEqual(tuple(value.shape),(2,))
        self.assertTrue(torch.isfinite(logits).all().item())
        saved=model_to_portable(net,'conv',metadata={'unit':'test'})
        self.assertEqual(saved['format'],'quartz_connect4_conv3x3_v1')
        self.assertEqual(len(saved['conv_weight']),16)
        self.assertEqual(len(saved['head_weight'][0]),672)
        legal=[True,True,False,True,False,True,True]
        pred,val=compare_portable(saved,inputs[0].tolist(),legal)
        expected=legal_masked_scores(logits[0].tolist(),legal)
        self.assertLess(max(abs(x-y) for x,y in zip(expected,pred)),3e-5)
        self.assertLess(abs(val-value[0].item()),3e-5)
        self.assertEqual(pred[2],0)
        self.assertEqual(pred[4],0)

    def test_group_split_identity_and_no_root_leakage(self):
        from json import loads
        folder=Path(__file__).resolve().parents[1]
        frozen=loads((folder/'COHORT_1024_FROZEN.json').read_text())
        info=validate_splits(frozen)
        self.assertEqual(info['families'],64)
        self.assertEqual(info['splits'],{'train':640,'val':192,'test':192})
        self.assertTrue(info['no_shared_family'])
        self.assertTrue(info['balanced_colors'])

    def test_illegal_policy_inputs_rejected(self):
        values=[0.0]*7
        with self.assertRaises(ValueError):
            legal_masked_scores(values,[False]*7)
        with self.assertRaises(ValueError):
            legal_masked_scores([float('nan')]*7,[True]*7)

if __name__=='__main__':unittest.main()
