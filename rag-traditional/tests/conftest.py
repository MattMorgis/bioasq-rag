"""Shared pytest fixtures for all rag-traditional tests."""

import sys
from pathlib import Path

# Add the project root to the Python path to enable proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fixtures can be added here as needed
