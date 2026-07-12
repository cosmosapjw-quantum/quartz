/**
 * @file interface.cpp
 * @brief Game adapter interface implementation
 *
 * This file provides the implementation for the unified interface for all
 * game implementations in the AlphaZero engine.
 */

#include <memory>
#include <unordered_map>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <functional>
#include <fstream>

#include "interface.h"
#include "chess/chess_state.h"
#include "go/go_state.h"
#include "gomoku/gomoku_state.h"

namespace alphazero {
namespace core {

// Utility functions for GameType conversion
std::string gameTypeToString(GameType type) {
    switch (type) {
        case GameType::CHESS: return "chess";
        case GameType::GO: return "go";
        case GameType::GOMOKU: return "gomoku";
        case GameType::UNKNOWN:
        default: return "unknown";
    }
}

GameType stringToGameType(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "chess") return GameType::CHESS;
    if (lower == "go") return GameType::GO;
    if (lower == "gomoku") return GameType::GOMOKU;
    return GameType::UNKNOWN;
}

// GameRegistry implementation
GameRegistry& GameRegistry::instance() {
    static GameRegistry instance;
    return instance;
}

void GameRegistry::registerGame(GameType type, GameFactoryFunction factory) {
    factories_[type] = std::move(factory);
}

bool GameRegistry::isRegistered(GameType type) const {
    return factories_.find(type) != factories_.end();
}

const GameFactoryFunction& GameRegistry::getFactory(GameType type) const {
    auto it = factories_.find(type);
    if (it == factories_.end()) {
        throw std::runtime_error("Game type " + gameTypeToString(type) + " is not registered");
    }
    return it->second;
}

std::vector<GameType> GameRegistry::getRegisteredTypes() const {
    std::vector<GameType> types;
    types.reserve(factories_.size());
    for (const auto& pair : factories_) {
        types.push_back(pair.first);
    }
    return types;
}

void GameRegistry::clear() {
    factories_.clear();
}

// GameFactory implementation
std::unique_ptr<IGameState> GameFactory::createGame(GameType type) {
    return GameRegistry::instance().getFactory(type)();
}

std::unique_ptr<IGameState> GameFactory::createGame(const std::string& typeName) {
    GameType type = stringToGameType(typeName);
    return createGame(type);
}

std::unique_ptr<IGameState> GameFactory::createChess(
    bool chess960,
    const std::string& fen,
    int position_number
) {
    return std::make_unique<games::chess::ChessState>(chess960, fen, position_number);
}

std::unique_ptr<IGameState> GameFactory::createGo(
    int board_size,
    int rule_set,
    float custom_komi
) {
    auto rule_enum = static_cast<games::go::GoState::RuleSet>(rule_set);
    return std::make_unique<games::go::GoState>(board_size, rule_enum, custom_komi);
}

std::unique_ptr<IGameState> GameFactory::createGomoku(
    int board_size,
    bool use_renju,
    bool use_omok,
    int seed,
    bool use_pro_long_opening
) {
    return std::make_unique<games::gomoku::GomokuState>(
        board_size, use_renju, use_omok, seed, use_pro_long_opening
    );
}

std::unique_ptr<IGameState> GameFactory::createGameFromMoves(
    GameType type,
    const std::string& moves
) {
    auto game = createGame(type);
    std::istringstream iss(moves);
    std::string move;

    while (iss >> move) {
        auto action = game->stringToAction(move);
        if (!action.has_value()) {
            throw std::runtime_error("Invalid move: " + move);
        }

        if (!game->isLegalMove(action.value())) {
            throw std::runtime_error("Illegal move: " + move);
        }

        game->makeMove(action.value());
    }

    return game;
}

std::vector<std::unique_ptr<IGameState>> GameFactory::createGames(
    GameType type,
    int count
) {
    std::vector<std::unique_ptr<IGameState>> games;
    games.reserve(count);
    const auto& factory = GameRegistry::instance().getFactory(type);
    for (int i = 0; i < count; ++i) {
        games.push_back(factory());
    }
    return games;
}

GameType GameFactory::detectGameType(const std::string& input) {
    std::string trimmed = input;
    // Simple trim
    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

    // Check for FEN (Chess)
    if (trimmed.find('/') != std::string::npos && trimmed.find(' ') != std::string::npos) {
        std::istringstream iss(trimmed);
        std::string part;
        int parts = 0;
        while (iss >> part && parts < 6) {
            parts++;
        }
        if (parts == 6) {
            return GameType::CHESS;
        }
    }

    // Check for SGF (Go)
    if (trimmed.find("(;") != std::string::npos || trimmed.find(";B[") != std::string::npos) {
        return GameType::GO;
    }

    // Check for chess notation patterns
    if (trimmed.length() == 4) {
        // e2e4 format
        if (trimmed[0] >= 'a' && trimmed[0] <= 'h' &&
            trimmed[1] >= '1' && trimmed[1] <= '8' &&
            trimmed[2] >= 'a' && trimmed[2] <= 'h' &&
            trimmed[3] >= '1' && trimmed[3] <= '8') {
            return GameType::CHESS;
        }
    }

    // Check for Go notation
    if (trimmed == "pass" || trimmed == "PASS") {
        return GameType::GO;
    }
    if (trimmed.length() >= 2) {
        if (trimmed[0] >= 'A' && trimmed[0] <= 'T' && std::isdigit(trimmed[1])) {
            return GameType::GO;
        }
    }

    // Check for Gomoku notation
    if (trimmed.length() >= 2) {
        if (trimmed[0] >= 'A' && trimmed[0] <= 'O' && std::isdigit(trimmed[1])) {
            return GameType::GOMOKU;
        }
    }

    return GameType::UNKNOWN;
}

// GameSerializer implementation
std::string GameSerializer::serializeGame(const IGameState& game) {
    std::ostringstream oss;

    // Header with game type
    oss << "GameType:" << gameTypeToString(game.getGameType()) << "\n";
    oss << "BoardSize:" << game.getBoardSize() << "\n";
    oss << "CurrentPlayer:" << game.getCurrentPlayer() << "\n";
    oss << "Terminal:" << (game.isTerminal() ? "true" : "false") << "\n";

    if (game.isTerminal()) {
        oss << "Result:" << static_cast<int>(game.getGameResult()) << "\n";
    }

    // Move history
    auto history = game.getMoveHistory();
    oss << "MoveHistory:";
    for (size_t i = 0; i < history.size(); ++i) {
        if (i > 0) oss << ",";
        oss << history[i];
    }
    oss << "\n";

    // Game-specific state
    oss << "State:" << game.toString() << "\n";
    oss << "Hash:" << std::hex << game.getHash() << std::dec << "\n";

    return oss.str();
}

std::unique_ptr<IGameState> GameSerializer::deserializeGame(const std::string& data) {
    std::istringstream iss(data);
    std::string line;
    std::unordered_map<std::string, std::string> fields;

    // Parse fields
    while (std::getline(iss, line)) {
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);
            fields[key] = value;
        }
    }

    // Get game type
    auto typeIt = fields.find("GameType");
    if (typeIt == fields.end()) {
        throw std::runtime_error("Missing GameType in serialized data");
    }

    GameType type = stringToGameType(typeIt->second);
    auto game = GameFactory::createGame(type);

    // Apply move history if present
    auto historyIt = fields.find("MoveHistory");
    if (historyIt != fields.end() && !historyIt->second.empty()) {
        std::istringstream moveStream(historyIt->second);
        std::string moveStr;

        while (std::getline(moveStream, moveStr, ',')) {
            if (!moveStr.empty()) {
                int action = std::stoi(moveStr);
                if (game->isLegalMove(action)) {
                    game->makeMove(action);
                }
            }
        }
    }

    return game;
}

void GameSerializer::saveGame(const IGameState& game, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << serializeGame(game);

    if (!file.good()) {
        throw std::runtime_error("Error writing to file: " + filename);
    }
}

std::unique_ptr<IGameState> GameSerializer::loadGame(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    if (!file.good() && !file.eof()) {
        throw std::runtime_error("Error reading from file: " + filename);
    }

    return deserializeGame(buffer.str());
}

std::string GameSerializer::exportToStandardFormat(const IGameState& game) {
    switch (game.getGameType()) {
        case GameType::CHESS:
            return exportChessToPGN(game);
        case GameType::GO:
            return exportGoToSGF(game);
        case GameType::GOMOKU:
            return exportGomokuToCustom(game);
        default:
            return serializeGame(game);
    }
}

std::string GameSerializer::exportChessToPGN(const IGameState& game) {
    // Basic PGN export
    std::ostringstream pgn;
    pgn << "[Event \"AlphaZero Game\"]\n";
    pgn << "[Result \"";

    if (game.isTerminal()) {
        auto result = game.getGameResult();
        switch (result) {
            case GameResult::WIN_PLAYER1: pgn << "1-0"; break;
            case GameResult::WIN_PLAYER2: pgn << "0-1"; break;
            case GameResult::DRAW: pgn << "1/2-1/2"; break;
            default: pgn << "*"; break;
        }
    } else {
        pgn << "*";
    }
    pgn << "\"]\n\n";

    // Move list (simplified)
    auto history = game.getMoveHistory();
    for (size_t i = 0; i < history.size(); ++i) {
        if (i % 2 == 0) {
            pgn << (i / 2 + 1) << ". ";
        }
        pgn << game.actionToString(history[i]) << " ";
    }

    return pgn.str();
}

std::string GameSerializer::exportGoToSGF(const IGameState& game) {
    std::ostringstream sgf;
    sgf << "(;FF[4]GM[1]SZ[" << game.getBoardSize() << "]";

    auto history = game.getMoveHistory();
    bool isBlack = true;

    for (int action : history) {
        sgf << ";" << (isBlack ? "B" : "W") << "[" << game.actionToString(action) << "]";
        isBlack = !isBlack;
    }

    sgf << ")";
    return sgf.str();
}

std::string GameSerializer::exportGomokuToCustom(const IGameState& game) {
    std::ostringstream custom;
    custom << "Gomoku Game\n";
    custom << "Board Size: " << game.getBoardSize() << "\n";
    custom << "Moves: ";

    auto history = game.getMoveHistory();
    for (size_t i = 0; i < history.size(); ++i) {
        if (i > 0) custom << " ";
        custom << game.actionToString(history[i]);
    }
    custom << "\n";
    custom << game.toString();

    return custom.str();
}

// Auto-register all built-in games
void initializeGameInterface() {
    static bool initialized = false;
    if (initialized) return;

    auto& registry = GameRegistry::instance();

    // Register Chess
    registry.registerGame(GameType::CHESS, []() {
        return std::make_unique<games::chess::ChessState>();
    });

    // Register Go
    registry.registerGame(GameType::GO, []() {
        return std::make_unique<games::go::GoState>();
    });

    // Register Gomoku
    registry.registerGame(GameType::GOMOKU, []() {
        return std::make_unique<games::gomoku::GomokuState>();
    });

    initialized = true;
}

// Static initialization to register games when module loads
namespace {
    struct GameInterfaceInitializer {
        GameInterfaceInitializer() {
            initializeGameInterface();
        }
    };
    static GameInterfaceInitializer init;
}

} // namespace core
} // namespace alphazero