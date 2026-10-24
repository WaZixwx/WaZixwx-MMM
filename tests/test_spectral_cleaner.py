"""
Regression tests for spectral cleaner performance safeguards.
"""

import unittest
try:
    import pytest
except ImportError:
    raise unittest.SkipTest('pytest is required for this test module')
try:
    import librosa
except ImportError:
    raise unittest.SkipTest('librosa is required for this test module')


import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "mmm" / "sanitization" / "spectral_cleaner.py"
)
SPEC = importlib.util.spec_from_file_location("spectral_cleaner", MODULE_PATH)
spectral_cleaner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(spectral_cleaner)
SpectralCleaner = spectral_cleaner.SpectralCleaner


def test_extract_verification_window_caps_length():
    """Verification windows should be capped for large inputs."""
    cleaner = SpectralCleaner(paranoid_mode=False)
    long_audio = np.random.randn(cleaner.max_verification_samples * 3, 2).astype(
        np.float32
    )

    window = cleaner._extract_verification_window(
        long_audio, cleaner.max_verification_samples
    )

    assert window.shape[0] == cleaner.max_verification_samples
    assert window.shape[1] == 2


def test_verify_cleaning_bounds_autocorrelation_input(monkeypatch):
    """Autocorrelation input must be bounded to avoid quadratic runtime."""
    cleaner = SpectralCleaner(paranoid_mode=False)
    seen_lengths = []
    original_correlate = spectral_cleaner.signal.correlate

    def spy_correlate(x, y, mode="same", method="auto"):
        seen_lengths.append(len(x))
        return original_correlate(x, y, mode=mode, method=method)

    monkeypatch.setattr(spectral_cleaner.signal, "correlate", spy_correlate)

    samples = cleaner.max_verification_samples * 5
    original = np.random.randn(samples, 2).astype(np.float32)
    cleaned = original.copy()

    cleaner._verify_cleaning(original, cleaned, sample_rate=44100)

    assert seen_lengths
    assert max(seen_lengths) <= cleaner.max_autocorr_samples


def test_disrupt_periodic_patterns_limits_notch_filter_count(monkeypatch):
    """Dense peak sets should still produce a bounded number of notch passes."""
    cleaner = SpectralCleaner(paranoid_mode=False)
    cleaner.max_notch_filters = 7
    notch_calls = []

    def fake_find_peaks(*_args, **_kwargs):
        peaks = np.arange(5, 105)
        heights = np.linspace(1.0, 2.0, len(peaks))
        return peaks, {"peak_heights": heights}

    def fake_notch(data, freq, sample_rate, q=30):
        notch_calls.append((freq, sample_rate, q))
        return data

    monkeypatch.setattr(spectral_cleaner.signal, "find_peaks", fake_find_peaks)
    monkeypatch.setattr(cleaner, "_apply_notch_filter", fake_notch)

    sample_rate = 44100
    audio = np.random.randn(sample_rate).astype(np.float32)
    cleaner._disrupt_periodic_patterns(audio, sample_rate)

    assert len(notch_calls) == cleaner.max_notch_filters
