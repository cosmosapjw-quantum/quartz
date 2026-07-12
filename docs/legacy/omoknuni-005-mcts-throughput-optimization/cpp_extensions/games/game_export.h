/**
 * @file game_export.h
 * @brief Game module export definitions
 *
 * Provides export definitions for the game adapter interface,
 * making the GameRegistry, GameFactory, and GameSerializer
 * classes available to the Python bindings and other modules.
 */

#ifndef ALPHAZERO_GAMES_EXPORT_H
#define ALPHAZERO_GAMES_EXPORT_H

#include "interface.h"

namespace alphazero {
namespace core {

// Re-export the main interface classes for external use
using GameRegistry = alphazero::core::GameRegistry;
using GameFactory = alphazero::core::GameFactory;
using GameSerializer = alphazero::core::GameSerializer;

} // namespace core
} // namespace alphazero

#endif // ALPHAZERO_GAMES_EXPORT_H