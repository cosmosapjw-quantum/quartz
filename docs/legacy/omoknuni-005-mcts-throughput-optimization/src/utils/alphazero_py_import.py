"""
Centralized AlphaZero Python bindings import utility.

This module provides a singleton pattern for importing the alphazero_py C++ bindings
to prevent pybind11 registration conflicts when the module is imported multiple times
during test runs.
"""

import sys
import os
from typing import Optional, Any

_alphazero_py_module: Optional[Any] = None
_import_attempted: bool = False
_games_available: bool = False

def get_alphazero_py() -> Optional[Any]:
    """
    Get the alphazero_py module using singleton pattern.

    Returns:
        The alphazero_py module if available, None otherwise.
    """
    global _alphazero_py_module, _import_attempted, _games_available

    if not _import_attempted:
        _import_attempted = True

        # Add build directory to path for testing
        # Get absolute path to build directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        build_path = os.path.join(script_dir, '..', '..', 'build', 'cpp_extensions', 'games')
        build_path = os.path.abspath(build_path)

        if build_path not in sys.path:
            sys.path.insert(0, build_path)

        try:
            # Check if module is already loaded to prevent re-registration
            if 'alphazero_py' in sys.modules:
                _alphazero_py_module = sys.modules['alphazero_py']
                _games_available = True
            else:
                import alphazero_py
                _alphazero_py_module = alphazero_py
                _games_available = True
        except ImportError as e:
            # Fallback for testing without compiled extensions
            _alphazero_py_module = None
            _games_available = False
            # For debugging: uncomment the next line to see import errors
            # print(f"Failed to import alphazero_py: {e}")
        except Exception as e:
            # Handle any other exceptions (like pybind11 registration errors)
            _alphazero_py_module = None
            _games_available = False
            # For debugging: uncomment the next line to see import errors
            # print(f"Exception importing alphazero_py: {e}")

    return _alphazero_py_module

def games_available() -> bool:
    """
    Check if the alphazero_py games module is available.

    Returns:
        True if games are available, False otherwise.
    """
    global _games_available
    get_alphazero_py()  # Ensure import has been attempted
    return _games_available

def require_alphazero_py() -> Any:
    """
    Get the alphazero_py module, raising ImportError if not available.

    Returns:
        The alphazero_py module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    module = get_alphazero_py()
    if module is None:
        raise ImportError("Cannot import alphazero_py module. Build may be required.")
    return module