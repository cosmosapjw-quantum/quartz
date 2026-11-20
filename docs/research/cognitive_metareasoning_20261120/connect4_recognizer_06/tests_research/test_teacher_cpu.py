"""Tests of exact-label interpretation and leak-free encoding (CPU-only)."""
import copy
import json
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teacher_cpu import parse_exact_label, encode_relative, mirror_encoding


class TeacherContractTests(unittest.TestCase):
    def test_winning_action_set_not_single_move(self):
        x=parse_exact_label(
            history="4433226223",scores=[16,15,15,15,16,15,15],
            legal=list(range(7)),
        )
        self.assertEqual(x["root_wdl"],1)
        self.assertEqual(x["optimal_actions"],[0,1,2,3,4,5,6])
        self.assertEqual(x["strict_fastest_win_actions"],[0,4])
        self.assertEqual(x["target_policy"],[1/7]*7)

    def test_draw_favored_over_losing_action(self):
        x=parse_exact_label("3545444242",[-8,-7,-6,-5,0,-4,-3],list(range(7)))
        self.assertEqual(x["root_wdl"],0)
        self.assertEqual(x["optimal_actions"],[4])
        self.assertEqual(x["target_policy"][4],1.0)

    def test_illegal_score_sentinel_and_invalid_label_rejected(self):
        x=parse_exact_label("1",[-1000,0,0,0,0,0,0],[1,2,3,4,5,6])
        self.assertEqual(x["optimal_actions"],[1,2,3,4,5,6])
        with self.assertRaisesRegex(ValueError,"sentinel"):
            parse_exact_label("1",[-1000,0,0,0,0,0,0],list(range(7)))
        with self.assertRaises(ValueError):
            parse_exact_label("1",[1,2],list(range(7)))

    def test_features_current_player_relative_and_mirrored(self):
        b=[0]*42
        b[0]=1;b[1]=-1
        a=encode_relative(b,1)
        self.assertEqual(len(a),126)
        self.assertEqual(a[0],1)
        self.assertEqual(a[42+1],1)
        self.assertEqual(a[84],1)
        neg=encode_relative(b,-1)
        self.assertEqual(neg[1],1)
        self.assertEqual(neg[42],1)
        self.assertEqual(neg[84],0)
        mirrored=mirror_encoding(a)
        self.assertEqual(mirrored[6],1)
        self.assertEqual(mirrored[42+5],1)
        self.assertEqual(mirror_encoding(mirrored),a)


if __name__=="__main__":
    unittest.main()
