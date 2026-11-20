"""COG07 live Rust MCTS scoring, including exact signed negative outcomes."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark07 import score_response, summarize_mode

class COG07MCTSScoringTests(unittest.TestCase):
    def setUp(self):
        self.exact=[4,-5,0,-1000,2,5,-3]
        self.history='444444'
        self.board=[0]*42
        for y in range(6):
            self.board[y*7+3]=1 if y%2==0 else -1
        self.steps=[]
        for budget,move,calls,us in ((8,1,4,800),(16,2,6,900),(32,4,8,900),(64,5,10,1000)):
            self.steps.append({'budget':budget,'root_visits':budget,'root_hash':'abc',
                               'best_action':move,'evaluator_calls':calls,
                               'additional_evaluator_calls':calls-(2 if budget==8 else {16:4,32:6,64:8}[budget]),
                               'step_wall_us':us})
        self.raw={'schema':'quartz_connect4_resident_mcts_v1',
                  'history':self.history,'eval_mode':'trained_conv3x3_cpu',
                  'root_hash':'abc','root_initial_evaluator_calls':2,
                  'board_bottom_up':self.board,
                  'legal_columns_0based':[0,1,2,4,5,6],
                  'model_load_wall_us':25,
                  'root_setup_wall_us':15,
                  'total_request_wall_us_including_fresh16':5000,
                  'fresh16':{'resident_match':True},
                  'steps':self.steps}
    def test_signed_game_utility_and_observed_cost(self):
        score=score_response(self.raw,self.exact,self.raw['board_bottom_up'],[0,1,2,4,5,6])
        self.assertEqual(score['utility'],[-1,0,1,1])
        self.assertEqual(score['regret'],[2,1,0,0])
        self.assertEqual(score['delta'],[1,1,0])
        self.assertEqual(score['cumulative_evaluations'],[4,6,8,10])
        self.assertEqual(score['estimated_end_to_end_us_at_budget'][0],25+15+800)
    def test_score_rejects_illegal_or_wrong_root(self):
        bad=dict(self.raw)
        bad['steps']=[dict(v) for v in self.steps]
        bad['steps'][0]['best_action']=3
        with self.assertRaisesRegex(ValueError,"illegal"):
            score_response(bad,self.exact,self.raw['board_bottom_up'],[0,1,2,4,5,6])
        bad=dict(self.raw)
        bad['steps']=[dict(v) for v in self.steps]
        bad['steps'][2]['root_hash']='different'
        with self.assertRaisesRegex(ValueError,"root"):
            score_response(bad,self.exact,self.raw['board_bottom_up'],[0,1,2,4,5,6])
    def test_mode_summary_preserves_negative_mean(self):
        one=score_response(self.raw,self.exact,self.raw['board_bottom_up'],[0,1,2,4,5,6])
        summary=summarize_mode([{'id':'r1','family':1,'score':one}])
        self.assertEqual(summary['n'],1)
        self.assertEqual(summary['by_budget']['8']['mean_WDL'],-1)
        self.assertEqual(summary['by_budget']['64']['mean_regret'],0)

if __name__=='__main__':unittest.main()
