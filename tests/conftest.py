"""
Shared pytest fixtures and hooks for mmm test suite.

Forces garbage collection between tests to prevent cumulative OOM
from large numpy arrays, librosa caches, and ThreadPoolExecutor
thread references that survive executor context exit.
"""

import gc

import pytest


@pytest.fixture(autouse=True)
def force_gc_between_tests():
    """Run the test, then force full garbage collection."""
    yield
    gc.collect()
