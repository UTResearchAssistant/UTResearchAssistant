"""Core application configuration and setup utilities.

Modules in this package are responsible for loading configuration
variables, initialising logging and providing other shared resources.
"""

from .config import Settings

__all__ = ["Settings"]
