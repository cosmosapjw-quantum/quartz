"""Small policy/value MLP CPU distillation and portable-weight contract."""
import json
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_cpu import TinyPolicyValue, model_to_json, predict_json, expand_train_mirror


class TinyPolicyValueTests(unittest.TestCase):
    def test_exported_f32_weights_match_cpu_forward(self):
        torch.manual_seed(44)
        m=TinyPolicyValue().eval()
        inp=torch.rand(126,dtype=torch.float32).reshape(1,126)
        legal=[True,True,True,False,True,False,True]
        portable=model_to_json(m,metadata={"source":"synthetic-unit-test"})
        self.assertEqual(len(portable["w1"]),64)
        self.assertEqual(len(portable["w2"]),8)
        with torch.inference_mode():
            logits,value=m(inp)
            masked=logits.masked_fill(~torch.tensor([legal]),-1e9)
            ref=torch.softmax(masked,dim=1).tolist()[0]
        prob,val=predict_json(portable,inp[0].tolist(),legal)
        self.assertTrue(max(abs(p-q) for p,q in zip(ref,prob))<1e-6)
        self.assertTrue(abs(value.item()-val)<1e-6)
        self.assertEqual(prob[3],0)
        self.assertEqual(prob[5],0)

    def test_train_mirror_matches_policy_columns_and_features(self):
        row={"id":"x","split":"train","history":"12","features":[0.0]*126,
             "target_policy":[1.0,0,0,0,0,0,0],"legal_columns":[0,1,2,3,4,5,6],
             "root_wdl":1}
        row["features"][0]=1
        a=expand_train_mirror([row])
        self.assertEqual(len(a),2)
        self.assertEqual(a[1]["features"][6],1)
        self.assertEqual(a[1]["target_policy"][6],1)
        self.assertEqual(a[0]["root_wdl"],a[1]["root_wdl"])
        with self.assertRaisesRegex(ValueError,"train"):
            expand_train_mirror([dict(row,split="val")])


if __name__=="__main__":
    unittest.main()
