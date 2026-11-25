// SPDX-License-Identifier: GPL-3.0-or-later
// Separate local evaluator-only process for Rapfi 250615. No search or teacher labels.
#pragma once
#include "game/board.h"
#include "eval/mix9svqnnue.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline int quartzEvalBridge(const char *weight)
{
    try {
        using Evaluation::mix9svq::Evaluator;
        Board board(15, CandidateRange::FULL_BOARD);
        Evaluator ev(15, FREESTYLE, Numa::DefaultNumaNodeId, weight, weight);
        std::vector<int> history;
        auto reset = [&] {
            board.newGame(FREESTYLE);
            ev.initEmptyBoard();
            history.clear();
        };
        reset();
        std::cout << "QZ_READY 250615 mix9svq freestyle15" << std::endl;
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line=="QUIT") return 0;
            if (line=="RESET") { reset(); std::cout<<"QZ_RESET"<<std::endl; continue; }
            std::istringstream in(line);
            std::string cmd, cells, extra;
            uint64_t id;
            if (!(in>>cmd>>id>>cells) || cmd!="EVAL" || cells.size()!=225 || (in>>extra))
                throw std::runtime_error("bad evaluator request");
            std::array<std::vector<int>,2> stones;
            for (int i=0;i<225;++i) {
                if(cells[i]=='1'||cells[i]=='2')stones[cells[i]-'1'].push_back(i);
                else if(cells[i]!='0')throw std::runtime_error("invalid board character");
            }
            if (stones[0].size()<stones[1].size() || stones[0].size()>stones[1].size()+1 ||
                stones[0].size()+stones[1].size()>=225)
                throw std::runtime_error("invalid alternating board counts");
            // Canonical placement order. Reuse prefix and undo suffix; the
            // NN input is absolute board and current player, not move-order planes.
            std::vector<int> next;
            for(size_t i=0;i<stones[0].size();++i){
                next.push_back(stones[0][i]);
                if(i<stones[1].size())next.push_back(stones[1][i]);
            }
            size_t keep=0;
            while(keep<history.size()&&keep<next.size()&&history[keep]==next[keep])++keep;
            while(history.size()>keep){
                int m=history.back();Pos pos(m%15,m/15);
                ev.beforeUndo(board,pos);board.undo(FREESTYLE);ev.afterUndo(board,pos);
                history.pop_back();
            }
            for(size_t i=keep;i<next.size();++i){
                int m=next[i];Pos pos(m%15,m/15);
                ev.beforeMove(board,pos);board.move(FREESTYLE,pos);ev.afterMove(board,pos);
                history.push_back(m);
            }
            for(int m=0;m<225;++m){
                Color c=board.get(Pos(m%15,m/15));
                char got=c==BLACK?'1':c==WHITE?'2':c==EMPTY?'0':'?';
                if(got!=cells[m])throw std::runtime_error("board reconstruction mismatch");
            }
            if(board.ply()!=int(next.size()) || board.sideToMove()!=(next.size()%2?WHITE:BLACK))
                throw std::runtime_error("side reconstruction mismatch");
            auto v=ev.evaluateValue(board,Evaluation::ACC_LEVEL_BEST);
            Evaluation::PolicyBuffer p(15);
            p.setComputeFlagForAllEmptyCell(board);
            ev.evaluatePolicy(board,p,Evaluation::ACC_LEVEL_BEST);
            p.applySoftmax(1.0f);
            if(!v.hasWinLossRate()||!v.hasDrawRate()||!std::isfinite(v.win())||
               !std::isfinite(v.loss())||!std::isfinite(v.draw()))
                throw std::runtime_error("invalid WDL");
            std::cout<<"QZ_RESULT "<<id<<" "<<next.size()<<" "<<(board.sideToMove()==BLACK?1:-1)
                     <<" "<<v.win()<<" "<<v.loss()<<" "<<v.draw();
            for(int m=0;m<225;++m){
                float q=cells[m]=='0'?p(m):0.0f;
                if(!std::isfinite(q)||q<0)throw std::runtime_error("invalid policy");
                std::cout<<" "<<q;
            }
            std::cout<<std::endl;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr<<"QZ_ERROR "<<e.what()<<std::endl;
        return 1;
    }
}
