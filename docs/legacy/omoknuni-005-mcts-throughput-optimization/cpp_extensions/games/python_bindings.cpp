// src/python/bindings.cpp
// Simplified Python bindings for game logic only
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <string>

#include "igamestate.h"
#include "game_export.h"

#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"

// #include "utils/attack_defense_module.h" // Removed - will be implemented in neural network tasks

namespace py = pybind11;

namespace alphazero {
namespace python {

// Convert C++ tensor representation to Python numpy arrays
py::array_t<float> tensorToNumpy(const std::vector<std::vector<std::vector<float>>>& tensor) {
    if (tensor.empty() || tensor[0].empty() || tensor[0][0].empty()) {
        return py::array_t<float>();
    }
    
    size_t channels = tensor.size();
    size_t height = tensor[0].size();
    size_t width = tensor[0][0].size();
    
    py::array_t<float> array({channels, height, width});
    py::buffer_info info = array.request();
    float* data = static_cast<float*>(info.ptr);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                data[c * height * width + h * width + w] = tensor[c][h][w];
            }
        }
    }
    
    return array;
}

// Game registration helper
void registerGames() {
    static bool registered = false;
    if (!registered) {
        // Register Chess
        core::GameRegistry::instance().registerGame(
            core::GameType::CHESS,
            []() { return std::make_unique<games::chess::ChessState>(); }
        );
        
        // Register Go
        core::GameRegistry::instance().registerGame(
            core::GameType::GO,
            []() { return std::make_unique<games::go::GoState>(); }
        );
        
        // Register Gomoku
        core::GameRegistry::instance().registerGame(
            core::GameType::GOMOKU,
            []() { return std::make_unique<games::gomoku::GomokuState>(); }
        );
        
        registered = true;
    }
}

// Module definition
PYBIND11_MODULE(alphazero_py, m) {
    m.doc() = "AlphaZero Python bindings - Game Logic Only";
    
    // Register games on module import
    registerGames();
    
    // Game types
    py::enum_<core::GameType>(m, "GameType")
        .value("UNKNOWN", core::GameType::UNKNOWN)
        .value("CHESS", core::GameType::CHESS)
        .value("GO", core::GameType::GO)
        .value("GOMOKU", core::GameType::GOMOKU)
        .export_values();
    
    // Game result
    py::enum_<core::GameResult>(m, "GameResult")
        .value("ONGOING", core::GameResult::ONGOING)
        .value("WIN_PLAYER1", core::GameResult::WIN_PLAYER1)
        .value("WIN_PLAYER2", core::GameResult::WIN_PLAYER2)
        .value("DRAW", core::GameResult::DRAW)
        .export_values();
    
    // Game interface
    py::class_<core::IGameState>(m, "IGameState")
        .def("get_legal_moves", [](const core::IGameState& state) {
            auto legal_moves_list = state.getLegalMoves();
            int action_space_size = state.getActionSpaceSize();

            // Return list of move indices (NOT boolean mask)
            std::vector<int> move_indices;
            move_indices.reserve(legal_moves_list.size());

            for (int move : legal_moves_list) {
                if (move >= 0 && move < action_space_size) {
                    move_indices.push_back(move);
                } else if (move == -1 && action_space_size > 0) {
                    // Handle Go pass move: map -1 to last index (board_size² for Go)
                    move_indices.push_back(action_space_size - 1);
                }
            }

            // Convert to numpy array of int32
            auto result = py::array_t<int>(move_indices.size());
            auto buf = result.mutable_unchecked<1>();
            for (size_t i = 0; i < move_indices.size(); i++) {
                buf(i) = move_indices[i];
            }

            return result;
        })
        .def("is_legal_move", [](const core::IGameState& state, int action) {
            int action_space_size = state.getActionSpaceSize();
            // Handle Go pass move: map last index back to -1
            if (action == action_space_size - 1) {
                // Check if this is actually a pass move by testing if -1 is legal
                if (state.isLegalMove(-1)) {
                    return true;
                }
            }
            return state.isLegalMove(action);
        })
        .def("make_move", [](core::IGameState& state, int action) -> uint64_t {
            // T024b: Zero-copy make_move returns undo token
            int action_space_size = state.getActionSpaceSize();
            // Handle Go pass move: map last index back to -1
            if (action == action_space_size - 1 && state.isLegalMove(-1)) {
                return state.make_move(static_cast<uint16_t>(-1));
            } else {
                return state.make_move(static_cast<uint16_t>(action));
            }
        })
        .def("unmake_move", [](core::IGameState& state, int action, uint64_t undo_token) {
            // T024b: Zero-copy unmake_move restores state
            int action_space_size = state.getActionSpaceSize();
            // Handle Go pass move: map last index back to -1
            if (action == action_space_size - 1 && state.isLegalMove(-1)) {
                state.unmake_move(static_cast<uint16_t>(-1), undo_token);
            } else {
                state.unmake_move(static_cast<uint16_t>(action), undo_token);
            }
        })
        .def("zobrist_hash", &core::IGameState::zobrist_hash)
        .def("apply_move_inplace", [](core::IGameState& state, int action) {
            int action_space_size = state.getActionSpaceSize();
            // Handle Go pass move: map last index back to -1
            if (action == action_space_size - 1 && state.isLegalMove(-1)) {
                state.makeMove(-1);
            } else {
                state.makeMove(action);
            }
        })
        .def("undo_move", &core::IGameState::undoMove)
        .def("is_terminal", &core::IGameState::isTerminal)
        .def("get_game_result", &core::IGameState::getGameResult)
        .def("get_current_player", &core::IGameState::getCurrentPlayer)
        .def("get_board_size", &core::IGameState::getBoardSize)
        .def("get_action_space_size", &core::IGameState::getActionSpaceSize)
        .def_property_readonly("action_space_size", &core::IGameState::getActionSpaceSize)
        .def("get_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getTensorRepresentation());
        })
        .def("get_basic_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getBasicTensorRepresentation());
        })
        .def("get_enhanced_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getEnhancedTensorRepresentation());
        })
        .def("extract_features", [](const core::IGameState& state) {
            return tensorToNumpy(state.getBasicTensorRepresentation());  // Use basic 7-channel representation for contract compatibility
        })
        .def("extract_features_to_buffer", [](const core::IGameState& state, py::array_t<float>& buffer) {
            // T007e: Direct feature extraction to pre-allocated buffer
            if (buffer.size() != state.get_num_feature_planes() * state.getBoardSize() * state.getBoardSize()) {
                throw std::invalid_argument("Buffer size mismatch: expected " +
                    std::to_string(state.get_num_feature_planes() * state.getBoardSize() * state.getBoardSize()) +
                    " but got " + std::to_string(buffer.size()));
            }
            state.extract_features_to_buffer(buffer.mutable_data());
        })
        .def("get_num_feature_planes", &core::IGameState::get_num_feature_planes)
        .def("get_hash", &core::IGameState::getHash)
        .def("action_to_string", &core::IGameState::actionToString)
        .def("string_to_action", &core::IGameState::stringToAction)
        .def("to_string", &core::IGameState::toString)
        .def("get_move_history", &core::IGameState::getMoveHistory)
        .def("clone", &core::IGameState::clone, py::call_guard<py::gil_scoped_release>())
        .def("copy", &core::IGameState::clone, py::call_guard<py::gil_scoped_release>())  // Alias for contract compatibility
        .def("batch_clone", &core::IGameState::batchClone, py::call_guard<py::gil_scoped_release>())
        .def("copy_from", &core::IGameState::copyFrom, py::call_guard<py::gil_scoped_release>());
    
    // Game factory
    m.def("create_game", [](core::GameType type) {
        return core::GameFactory::createGame(type);
    });
    
    m.def("create_game_from_moves", [](core::GameType type, const std::string& moves) {
        return core::GameFactory::createGameFromMoves(type, moves);
    });
    
    // Game serialization
    m.def("save_game", &core::GameSerializer::saveGame);
    m.def("load_game", &core::GameSerializer::loadGame);
    m.def("serialize_game", &core::GameSerializer::serializeGame);
    m.def("deserialize_game", &core::GameSerializer::deserializeGame);
    
    // Utility functions
    m.def("game_type_to_string", &core::gameTypeToString);
    m.def("string_to_game_type", &core::stringToGameType);
    
    // Game-specific classes with all options exposed
    
    // Chess with chess960 support
    py::class_<games::chess::ChessState, core::IGameState>(m, "ChessState")
        .def(py::init<>())
        .def(py::init<bool>(), py::arg("chess960") = false)
        .def(py::init<bool, const std::string&>(), py::arg("chess960") = false, py::arg("fen") = "")
        .def(py::init<bool, const std::string&, int>(), 
             py::arg("chess960") = false, py::arg("fen") = "", py::arg("position_number") = -1)
        .def(py::init<const games::chess::ChessState&>());

    // Go with rule set options
    py::enum_<games::go::GoState::RuleSet>(m, "GoRuleSet")
        .value("CHINESE", games::go::GoState::RuleSet::CHINESE)
        .value("JAPANESE", games::go::GoState::RuleSet::JAPANESE)
        .value("KOREAN", games::go::GoState::RuleSet::KOREAN)
        .export_values();
        
    py::class_<games::go::GoState, core::IGameState>(m, "GoState")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("board_size") = 19)
        .def(py::init<int, float, bool, bool>(), 
             py::arg("board_size") = 19, py::arg("komi") = 7.5f, 
             py::arg("chinese_rules") = true, py::arg("enforce_superko") = true)
        .def(py::init<int, games::go::GoState::RuleSet, float>(),
             py::arg("board_size"), py::arg("rule_set"), py::arg("custom_komi") = -1.0f);

    // Gomoku with all rule variants
    py::class_<games::gomoku::GomokuState, core::IGameState>(m, "GomokuState")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("board_size") = 15)
        .def(py::init<int, bool, bool, int, bool>(),
             py::arg("board_size") = 15, py::arg("use_renju") = false,
             py::arg("use_omok") = false, py::arg("seed") = 0,
             py::arg("use_pro_long_opening") = false)
        .def("get_renju_rules", &games::gomoku::GomokuState::getRenjuRules)
        .def("get_omok_rules", &games::gomoku::GomokuState::getOmokRules)
        .def("get_pro_long_opening", &games::gomoku::GomokuState::getProLongOpening);

}

} // namespace python
} // namespace alphazero