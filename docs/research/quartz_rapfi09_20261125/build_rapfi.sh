#!/usr/bin/env bash
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
P=/home/cosmosapjw/quantum/research
U="$P/rapfi_250615_quartz_bridge_20261125"
T="$P/rapfi_250615_quartz_bridge_build_20261125"
test "$(git -C "$U" rev-parse HEAD)" = 1be1551ced57e38d53ed58f6d74bf6f8b4bdc230
test ! -e "$T"
cp "$D/quartz_eval_bridge.h" "$U/Rapfi/quartz_eval_bridge.h"
python3 - "$U/Rapfi/main.cpp" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
a='#include "command/command.h"'
assert s.count(a)==1
s=s.replace(a,a+'\n#include "quartz_eval_bridge.h"')
a='    Command::CommandLine::init(argc, argv);'
assert s.count(a)==1
s=s.replace(a,a+'''
    if (argc == 3 && std::string(argv[1]) == "--quartz-eval") {
        if (!Command::loadConfig()) return EXIT_FAILURE;
        int rc=quartzEvalBridge(argv[2]);
        Search::Threads.waitForIdle();
        Search::Threads.setNumThreads(0);
        return rc;
    }
''')
p.write_text(s)
PY
cmake -S "$U/Rapfi" -B "$T" -DCMAKE_BUILD_TYPE=Release -DNO_COMMAND_MODULES=ON -DUSE_AVX2=ON -DUSE_SSE=ON -DUSE_AVX512=OFF -DUSE_VNNI=OFF -DENABLE_LTO=OFF -DCMAKE_CXX_COMPILER=clang++
nice -n 10 cmake --build "$T" -j 6
test -x "$T/pbrain-rapfi"
sha256sum "$T/pbrain-rapfi"
