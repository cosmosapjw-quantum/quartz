# C1. Source baseline and validation bounds

Locked QUARTZ R10 parent 617692a7ed85, 15x15 Freestyle primary rule, Rapfi bundled Linux AVX2 pretrained NNUE as independent alpha-beta teacher. Explicitly not a QUARTZ ONNX evaluator or exact full-game oracle.

Root action range 0..224, initial top16, at most4 reopening, first8 then32 total visits. Freeze 12 natural and 12 tactical-stress source positions before Rapfi teacher. Separate Renju rule4 legality from Freestyle rule0; assert actual legal cells.

The immutable baseline source and byte identities are in C01_INPUT_LOCK.json. Physical CPU compatibility requires measured wall time, not same visit count. All model files remain outside distributed Git artifact archives.
