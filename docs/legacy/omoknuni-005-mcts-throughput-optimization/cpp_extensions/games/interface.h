/**
 * @file interface.h
 * @brief Game adapter interface header
 *
 * This header provides the unified interface for all game implementations
 * in the AlphaZero engine, including factory pattern, registry, and
 * serialization capabilities.
 */

#ifndef ALPHAZERO_GAMES_INTERFACE_H
#define ALPHAZERO_GAMES_INTERFACE_H

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <fstream>

#include "../utils/igamestate.h"
#include "../utils/export_macros.h"

namespace alphazero {
namespace core {

// Forward declarations
class GameRegistry;
class GameFactory;
class GameSerializer;

/**
 * @brief Game factory function type
 */
using GameFactoryFunction = std::function<std::unique_ptr<IGameState>()>;

/**
 * @brief Singleton registry for game types and their factories
 *
 * This registry allows runtime registration of game types and provides
 * a centralized way to create game instances without tight coupling.
 */
class ALPHAZERO_API GameRegistry {
public:
    /**
     * @brief Get the singleton instance
     */
    static GameRegistry& instance();

    /**
     * @brief Register a game type with its factory function
     *
     * @param type Game type to register
     * @param factory Factory function that creates new instances
     */
    void registerGame(GameType type, GameFactoryFunction factory);

    /**
     * @brief Check if a game type is registered
     *
     * @param type Game type to check
     * @return true if registered, false otherwise
     */
    bool isRegistered(GameType type) const;

    /**
     * @brief Get the factory function for a game type
     *
     * @param type Game type
     * @return Factory function
     * @throws std::runtime_error if type is not registered
     */
    const GameFactoryFunction& getFactory(GameType type) const;

    /**
     * @brief Get all registered game types
     *
     * @return Vector of registered game types
     */
    std::vector<GameType> getRegisteredTypes() const;

    /**
     * @brief Clear all registrations (mainly for testing)
     */
    void clear();

private:
    GameRegistry() = default;
    std::unordered_map<GameType, GameFactoryFunction> factories_;
};

/**
 * @brief Factory for creating game instances
 *
 * Provides static methods to create game instances using the registry
 * or directly with specific parameters.
 */
class ALPHAZERO_API GameFactory {
public:
    /**
     * @brief Create a game instance of the specified type
     *
     * @param type Game type to create
     * @return Unique pointer to the game instance
     * @throws std::runtime_error if type is not registered
     */
    static std::unique_ptr<IGameState> createGame(GameType type);

    /**
     * @brief Create a game instance from a string type name
     *
     * @param typeName String representation of game type
     * @return Unique pointer to the game instance
     * @throws std::runtime_error if type name is invalid or not registered
     */
    static std::unique_ptr<IGameState> createGame(const std::string& typeName);

    /**
     * @brief Create a chess game with specific options
     *
     * @param chess960 Whether to use Chess960 rules
     * @param fen Optional FEN string for initial position
     * @param position_number Chess960 position number (0-959) if applicable
     * @return Unique pointer to chess game instance
     */
    static std::unique_ptr<IGameState> createChess(
        bool chess960 = false,
        const std::string& fen = "",
        int position_number = -1
    );

    /**
     * @brief Create a Go game with specific options
     *
     * @param board_size Board size (9, 13, or 19)
     * @param rule_set Rule set to use
     * @param custom_komi Optional custom komi value
     * @return Unique pointer to Go game instance
     */
    static std::unique_ptr<IGameState> createGo(
        int board_size = 19,
        int rule_set = 0,  // 0=Chinese, 1=Japanese, 2=Korean
        float custom_komi = -1.0f
    );

    /**
     * @brief Create a Gomoku game with specific options
     *
     * @param board_size Board size (typically 15)
     * @param use_renju Whether to use Renju rules
     * @param use_omok Whether to use Omok rules
     * @param seed Random seed for initialization
     * @param use_pro_long_opening Whether to use pro-long opening restrictions
     * @return Unique pointer to Gomoku game instance
     */
    static std::unique_ptr<IGameState> createGomoku(
        int board_size = 15,
        bool use_renju = false,
        bool use_omok = false,
        int seed = 0,
        bool use_pro_long_opening = false
    );

    /**
     * @brief Create a game instance from a sequence of moves
     *
     * @param type Game type
     * @param moves String containing move sequence (format depends on game)
     * @return Unique pointer to game instance with moves applied
     * @throws std::runtime_error if moves are invalid
     */
    static std::unique_ptr<IGameState> createGameFromMoves(
        GameType type,
        const std::string& moves
    );

    /**
     * @brief Create multiple game instances efficiently
     *
     * @param type Game type
     * @param count Number of instances to create
     * @return Vector of game instances
     */
    static std::vector<std::unique_ptr<IGameState>> createGames(
        GameType type,
        int count
    );

    /**
     * @brief Detect game type from state or move notation
     *
     * @param input String containing game state or moves
     * @return Detected game type
     */
    static GameType detectGameType(const std::string& input);
};

/**
 * @brief Game state serialization and deserialization
 *
 * Handles saving and loading game states to/from various formats.
 */
class ALPHAZERO_API GameSerializer {
public:
    /**
     * @brief Serialize a game state to string
     *
     * @param game Game state to serialize
     * @return Serialized string representation
     */
    static std::string serializeGame(const IGameState& game);

    /**
     * @brief Deserialize a game state from string
     *
     * @param data Serialized string representation
     * @return Unique pointer to deserialized game state
     * @throws std::runtime_error if deserialization fails
     */
    static std::unique_ptr<IGameState> deserializeGame(const std::string& data);

    /**
     * @brief Save game state to file
     *
     * @param game Game state to save
     * @param filename Output filename
     * @throws std::runtime_error if file cannot be written
     */
    static void saveGame(const IGameState& game, const std::string& filename);

    /**
     * @brief Load game state from file
     *
     * @param filename Input filename
     * @return Unique pointer to loaded game state
     * @throws std::runtime_error if file cannot be read or parsed
     */
    static std::unique_ptr<IGameState> loadGame(const std::string& filename);

    /**
     * @brief Export game to standard format (PGN for Chess, SGF for Go, custom for Gomoku)
     *
     * @param game Game state to export
     * @return String in standard format
     */
    static std::string exportToStandardFormat(const IGameState& game);

private:
    static std::string exportChessToPGN(const IGameState& game);
    static std::string exportGoToSGF(const IGameState& game);
    static std::string exportGomokuToCustom(const IGameState& game);
};

/**
 * @brief Initialize the game interface system
 *
 * Registers all built-in game types with their factory functions.
 * This is called automatically when the module loads.
 */
ALPHAZERO_API void initializeGameInterface();

/**
 * @brief Game adapter utilities
 */
namespace game_adapter {

/**
 * @brief Check if two game states are equivalent
 *
 * @param state1 First game state
 * @param state2 Second game state
 * @return true if equivalent, false otherwise
 */
ALPHAZERO_API bool areStatesEquivalent(const IGameState& state1, const IGameState& state2);

/**
 * @brief Get game statistics
 *
 * @param game Game state to analyze
 * @return Map of statistic names to values
 */
ALPHAZERO_API std::unordered_map<std::string, double> getGameStatistics(const IGameState& game);

/**
 * @brief Validate a sequence of moves
 *
 * @param game Initial game state
 * @param moves Vector of actions to validate
 * @return true if all moves are legal in sequence, false otherwise
 */
ALPHAZERO_API bool validateMoveSequence(IGameState& game, const std::vector<int>& moves);

/**
 * @brief Convert between different action representations
 *
 * @param game Game state for context
 * @param action Action to convert
 * @param format Target format ("string", "coordinate", "index")
 * @return Converted action representation
 */
ALPHAZERO_API std::string convertActionFormat(
    const IGameState& game,
    int action,
    const std::string& format
);

/**
 * @brief Get game complexity metrics
 *
 * @param type Game type
 * @return Map of complexity metrics (branching factor, game length, etc.)
 */
ALPHAZERO_API std::unordered_map<std::string, double> getGameComplexity(GameType type);

} // namespace game_adapter

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_GAMES_INTERFACE_H