import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rapfi_teacher14 import protocol_for_history,parse_move_and_model

class RapfiTeacherAdapterTests(unittest.TestCase):
 def test_freestyle_board_coordinate_and_side_parity(self):
  wire=protocol_for_history([31,160,32,177],300,0)
  self.assertIn('START 15\n',wire)
  self.assertIn('INFO rule 0\n',wire)
  self.assertIn('INFO timeout_turn 300\n',wire)
  self.assertIn('1,2,1\n',wire)
  self.assertIn('10,10,2\n',wire)
  self.assertIn('2,2,1\n',wire)
  self.assertTrue(wire.endswith('DONE\n'))
 def test_proven_model_and_move_not_log_line(self):
  answer='OK\nMESSAGE mix9svq nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4\n7,7\n'
  parsed=parse_move_and_model(answer,rule=0)
  self.assertEqual(parsed['xy'],[7,7])
  self.assertEqual(parsed['flat_action'],112)
  self.assertTrue(parsed['compatible_weight_reported'])
 def test_error_is_not_silent_draw(self):
  with self.assertRaises(ValueError):
   parse_move_and_model('ERROR model load failed\n',rule=0)
  with self.assertRaises(ValueError):
   protocol_for_history([225],300,0)
if __name__=='__main__':unittest.main()
