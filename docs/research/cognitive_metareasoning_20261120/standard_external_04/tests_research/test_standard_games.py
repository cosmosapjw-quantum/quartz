"""Standard-game external engine contract tests (no external subprocess in unit tests)."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standard_games import (
    c4_from_sequence, c4_has_four, c4_action_scores,
    gomoku_has_five, rapfi_parse_move,
    generate_c4_positions, generate_gomoku_positions,
)


class StandardGamesTests(unittest.TestCase):
    def test_c4_gravity_column_count_and_parity(self):
        b=c4_from_sequence("1212")
        self.assertEqual(len(b),42)
        self.assertEqual(b[0],1)
        self.assertEqual(b[1],-1)
        self.assertEqual(b[7],1)
        self.assertEqual(b[8],-1)

    def test_c4_terminal_history_rejected(self):
        with self.assertRaisesRegex(ValueError,"terminal"):
            c4_from_sequence("1212121")

    def test_c4_diagonal_and_floating_not_wins(self):
        b=[0]*42
        for k in [0,8,16,24]: b[k]=1
        self.assertTrue(c4_has_four(b,1))
        c=[0]*42
        for k in [0,14,21,28]:c[k]=1
        self.assertFalse(c4_has_four(c,1))

    def test_c4_solver_per_action_score_and_illegal(self):
        sequence="444333"
        b=c4_from_sequence(sequence)
        answer=c4_action_scores(sequence,
            "444333 -2 -2 1 3 -2 -2 -2",b)
        self.assertEqual(len(answer),7)
        self.assertEqual(answer[3],3)
        self.assertEqual(answer[0],-2)
        self.assertEqual(answer[6],-2)
        with self.assertRaisesRegex(ValueError,"sequence"):
            c4_action_scores(sequence,"222333 0 0 0 0 0 0 0",b)

    def test_gomoku_five_is_not_four(self):
        b=[0]*225
        for col in range(5):
            b[7*15+col]=1
        self.assertTrue(gomoku_has_five(b,1))
        self.assertFalse(gomoku_has_five(b,-1))
        b[7*15+4]=0
        self.assertFalse(gomoku_has_five(b,1))

    def test_rapfi_coord_validation(self):
        b=[0]*225
        b[7*15+7]=1
        p=rapfi_parse_move("MESSAGE Load config\nOK\nMESSAGE mix9svq nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4\n7,6\n",b)
        self.assertEqual(p,6*15+7)
        with self.assertRaisesRegex(ValueError,"illegal"):
            rapfi_parse_move("OK\nMESSAGE nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4\n7,7\n",b)
        with self.assertRaisesRegex(ValueError,"NNUE|weight"):
            rapfi_parse_move("OK\n6,6\n",b)

    def test_frozen_generation_is_deterministic_and_bounded(self):
        c=generate_c4_positions(seed=20261120)
        d=generate_c4_positions(seed=20261120)
        self.assertEqual(c,d)
        self.assertEqual(len(c),12)
        self.assertEqual(sorted([p["plies"] for p in c]),[10]*3+[14]*3+[18]*3+[22]*3)
        for p in c:
            b=c4_from_sequence(p["sequence"])
            self.assertFalse(c4_has_four(b,1) or c4_has_four(b,-1))
            self.assertEqual(len(p["sequence"]),p["plies"])
        g=generate_gomoku_positions(seed=20261120)
        self.assertEqual(g,generate_gomoku_positions(seed=20261120))
        self.assertEqual(len(g),12)
        self.assertEqual(sorted([x["plies"] for x in g]),[4]*3+[6]*3+[8]*3+[10]*3)
        for p in g:
            self.assertEqual(len(p["board"]),225)
            self.assertEqual(sum(abs(z) for z in p["board"]),p["plies"])
            self.assertFalse(gomoku_has_five(p["board"],1) or gomoku_has_five(p["board"],-1))
            self.assertEqual(p["player"],1)


if __name__=="__main__":
    unittest.main()
