"""
Tests for preserving_sanitizer module
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


import numpy as np
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace
import soundfile as sf

import mmm.preserving_sanitizer as preserving_module
from mmm.preserving_sanitizer import (
    preserving_sanitize,
    _ensure_channel_layout,
    _gentle_spectral_phase_noise,
    _add_hf_noise_and_dither,
    _apply_humanization,
    _apply_micro_resample_warp,
    _apply_resample_nudge,
    _apply_rms_gated_resample_nudge,
    _apply_analog_warmth,
    _apply_gentle_bandlimit,
    _add_micro_ambience,
    _apply_clarity_tilt,
    _apply_phase_swirl,
    _apply_phase_noise_fft,
    _apply_subblock_phase_dither,
    _apply_dynamic_comb_mask,
    _apply_transient_micro_shift,
    _apply_micro_eq_modulation,
    _apply_mfcc_perturbation,
    _repair_non_finite_audio,
)


class TestPreservingSanitizer:
    """Test cases for preserving sanitizer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 44100
        self.duration = 0.5  # Short duration for fast tests

        # Create test audio data
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.mono_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        self.stereo_audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 880 * t)
        ]).astype(np.float32)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str, audio_data: np.ndarray = None) -> Path:
        """Create a test audio file"""
        if audio_data is None:
            audio_data = self.mono_audio
        file_path = self.test_dir / filename
        sf.write(str(file_path), audio_data, self.sample_rate)
        return file_path


class TestEnsureChannelLayout:
    """Test _ensure_channel_layout helper"""

    def test_mono_1d_to_2d(self):
        """Test converting mono 1D to 2D"""
        mono = np.array([1.0, 2.0, 3.0, 4.0])
        result = _ensure_channel_layout(mono)
        assert result.ndim == 2
        assert result.shape == (4, 1)

    def test_stereo_correct_shape(self):
        """Test stereo with correct shape (samples, channels)"""
        stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        result = _ensure_channel_layout(stereo)
        assert result.shape == (3, 2)

    def test_stereo_transposed(self):
        """Test stereo with transposed shape (channels, samples)"""
        stereo = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # (2, 4)
        result = _ensure_channel_layout(stereo)
        assert result.shape == (4, 2)

    def test_none_input(self):
        """Test None input returns None"""
        result = _ensure_channel_layout(None)
        assert result is None


class TestGentleSpectralPhaseNoise:
    """Test _gentle_spectral_phase_noise helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 880 * t)
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _gentle_spectral_phase_noise(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape

    def test_paranoid_mode_difference(self):
        """Test paranoid mode produces different result"""
        normal = _gentle_spectral_phase_noise(self.audio.copy(), self.sample_rate, False)
        paranoid = _gentle_spectral_phase_noise(self.audio.copy(), self.sample_rate, True)
        # Due to randomness, they should differ
        assert not np.allclose(normal, paranoid)

    def test_audio_not_destroyed(self):
        """Test audio is not completely destroyed"""
        result = _gentle_spectral_phase_noise(self.audio, self.sample_rate, False)
        # Correlation should still be high
        corr = np.corrcoef(self.audio[:, 0].flatten(), result[:, 0].flatten())[0, 1]
        assert corr > 0.5  # Should maintain some similarity


class TestAddHFNoiseAndDither:
    """Test _add_hf_noise_and_dither helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _add_hf_noise_and_dither(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape

    def test_adds_noise(self):
        """Test that noise is added"""
        result = _add_hf_noise_and_dither(self.audio, self.sample_rate, False)
        diff = np.abs(result - self.audio)
        assert np.any(diff > 0)  # Some difference should exist

    def test_noise_is_subtle(self):
        """Test noise is subtle (not destructive)"""
        result = _add_hf_noise_and_dither(self.audio, self.sample_rate, False)
        max_diff = np.max(np.abs(result - self.audio))
        assert max_diff < 0.01  # Should be very small


class TestApplyHumanization:
    """Test _apply_humanization helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.stereo_audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 880 * t)
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_humanization(self.stereo_audio, self.sample_rate, False)
        assert result.shape == self.stereo_audio.shape

    def test_stereo_decorrelation(self):
        """Test stereo channels become decorrelated"""
        result = _apply_humanization(self.stereo_audio, self.sample_rate, False)
        # Channels should be slightly different after decorrelation
        assert not np.allclose(result[:, 0], result[:, 1])


class TestApplyMicroResampleWarp:
    """Test _apply_micro_resample_warp helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_micro_resample_warp(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape

    def test_audio_modified(self):
        """Test audio is modified"""
        result = _apply_micro_resample_warp(self.audio, self.sample_rate, False)
        assert not np.allclose(result, self.audio)


class TestApplyResampleNudge:
    """Test _apply_resample_nudge helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_resample_nudge(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestApplyAnalogWarmth:
    """Test _apply_analog_warmth helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_analog_warmth(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape

    def test_soft_saturation_applied(self):
        """Test soft saturation is applied (values clamped)"""
        # Create audio with peaks
        loud_audio = self.audio * 2.0
        result = _apply_analog_warmth(loud_audio, self.sample_rate, False)
        # After soft saturation, max should be reduced
        assert np.max(np.abs(result)) < np.max(np.abs(loud_audio))


class TestApplyGentleBandlimit:
    """Test _apply_gentle_bandlimit helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_gentle_bandlimit(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestAddMicroAmbience:
    """Test _add_micro_ambience helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.stereo_audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 880 * t)
        ]).astype(np.float32)
        self.mono_audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_stereo_output_shape_preserved(self):
        """Test stereo output shape matches input"""
        result = _add_micro_ambience(self.stereo_audio, self.sample_rate, False)
        assert result.shape == self.stereo_audio.shape

    def test_mono_output_shape_preserved(self):
        """Test mono output shape matches input"""
        result = _add_micro_ambience(self.mono_audio, self.sample_rate, False)
        assert result.shape == self.mono_audio.shape


class TestApplyClarityTilt:
    """Test _apply_clarity_tilt helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_clarity_tilt(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestApplyPhaseSwirl:
    """Test _apply_phase_swirl helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_phase_swirl(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestApplyPhaseNoiseFft:
    """Test _apply_phase_noise_fft helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_phase_noise_fft(self.audio, False)
        assert result.shape == self.audio.shape


class TestApplySubblockPhaseDither:
    """Test _apply_subblock_phase_dither helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_subblock_phase_dither(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestApplyDynamicCombMask:
    """Test _apply_dynamic_comb_mask helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_dynamic_comb_mask(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestApplyTransientMicroShift:
    """Test _apply_transient_micro_shift helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        # Create audio with clear transients
        audio = np.zeros_like(t)
        audio[1000:1100] = 0.5  # Transient
        audio[5000:5100] = 0.5  # Another transient
        self.audio = np.column_stack([audio]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_transient_micro_shift(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestApplyMicroEqModulation:
    """Test _apply_micro_eq_modulation helper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        self.audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
        ]).astype(np.float32)

    def test_output_shape_preserved(self):
        """Test output shape matches input"""
        result = _apply_micro_eq_modulation(self.audio, self.sample_rate, False)
        assert result.shape == self.audio.shape


class TestMfccPerturbationSafety:
    """Safety tests for MFCC perturbation numeric stability."""

    def test_repair_non_finite_audio(self):
        """NaN/Inf values are replaced with finite clipped samples."""
        audio = np.array([[np.nan, np.inf], [-np.inf, 0.5]], dtype=np.float32)
        repaired = _repair_non_finite_audio(audio, label="unit-test")

        assert np.all(np.isfinite(repaired))
        assert np.max(np.abs(repaired)) <= 1.0

    def test_mfcc_perturbation_fallback_on_reconstruction_failure(self, monkeypatch):
        """MFCC perturbation should keep original channel when inverse fails."""
        n = 44100
        audio = np.column_stack(
            [0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / 44100)]
        ).astype(np.float32)

        fake_inverse = SimpleNamespace(
            mfcc_to_mel=lambda mfccs, n_mels=128: np.ones((n_mels, mfccs.shape[1])),
            mel_to_audio=lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("forced inverse failure")
            ),
        )
        fake_feature = SimpleNamespace(
            melspectrogram=lambda **kwargs: np.ones((128, 12), dtype=np.float32),
            mfcc=lambda S, n_mfcc=13: np.ones((n_mfcc, S.shape[1]), dtype=np.float32),
            inverse=fake_inverse,
        )
        fake_librosa = SimpleNamespace(
            feature=fake_feature,
            power_to_db=np.zeros_like,
            db_to_power=np.ones_like,
        )

        monkeypatch.setattr(preserving_module, "librosa", fake_librosa)
        result = _apply_mfcc_perturbation(audio.copy(), 44100, paranoid_mode=False)

        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))
        assert np.allclose(result, audio)

    def test_mfcc_perturbation_fallback_on_non_finite_reconstruction(self, monkeypatch):
        """Non-finite reconstructed audio should be rejected and fallback to source."""
        n = 44100
        audio = np.column_stack(
            [0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / 44100)]
        ).astype(np.float32)

        bad_reconstruction = np.full(n, np.inf, dtype=np.float32)
        fake_inverse = SimpleNamespace(
            mfcc_to_mel=lambda mfccs, n_mels=128: np.ones((n_mels, mfccs.shape[1])),
            mel_to_audio=lambda *args, **kwargs: bad_reconstruction,
        )
        fake_feature = SimpleNamespace(
            melspectrogram=lambda **kwargs: np.ones((128, 12), dtype=np.float32),
            mfcc=lambda S, n_mfcc=13: np.ones((n_mfcc, S.shape[1]), dtype=np.float32),
            inverse=fake_inverse,
        )
        fake_librosa = SimpleNamespace(
            feature=fake_feature,
            power_to_db=np.zeros_like,
            db_to_power=np.ones_like,
        )

        monkeypatch.setattr(preserving_module, "librosa", fake_librosa)
        result = _apply_mfcc_perturbation(audio.copy(), 44100, paranoid_mode=True)

        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))
        assert np.allclose(result, audio)


class TestPreservingSanitizeIntegration:
    """Integration tests for preserving_sanitize function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 44100
        self.duration = 0.5

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str) -> Path:
        """Create a test audio file"""
        file_path = self.test_dir / filename
        sf.write(str(file_path), self.test_audio, self.sample_rate)
        return file_path

    def test_basic_sanitization(self):
        """Test basic sanitization works"""
        input_file = self.create_test_audio_file("input.wav")
        output_file = self.test_dir / "output.wav"

        result = preserving_sanitize(
            input_file,
            output_file,
            paranoid_mode=False,
            threat_count=0
        )

        assert result["success"] is True
        assert Path(result["output_file"]).exists()
        assert "stats" in result

    def test_paranoid_mode(self):
        """Test paranoid mode sanitization"""
        input_file = self.create_test_audio_file("input_paranoid.wav")
        output_file = self.test_dir / "output_paranoid.wav"

        result = preserving_sanitize(
            input_file,
            output_file,
            paranoid_mode=True,
            threat_count=10
        )

        assert result["success"] is True
        assert result["stats"]["processing_time"] > 0

    def test_all_stealth_flags_disabled(self):
        """Test with all stealth flags disabled"""
        input_file = self.create_test_audio_file("input_no_stealth.wav")
        output_file = self.test_dir / "output_no_stealth.wav"

        result = preserving_sanitize(
            input_file,
            output_file,
            paranoid_mode=False,
            phase_dither=False,
            comb_mask=False,
            transient_shift=False,
            resample_nudge=False,
            phase_noise=False,
            phase_swirl=False
        )

        assert result["success"] is True

    def test_all_stealth_flags_enabled(self):
        """Test with all stealth flags enabled"""
        input_file = self.create_test_audio_file("input_all_stealth.wav")
        output_file = self.test_dir / "output_all_stealth.wav"

        result = preserving_sanitize(
            input_file,
            output_file,
            paranoid_mode=True,
            phase_dither=True,
            comb_mask=True,
            transient_shift=True,
            resample_nudge=True,
            gated_resample_nudge=True,
            phase_noise=True,
            phase_swirl=True,
            masked_hf_phase=True,
            micro_eq_flutter=True,
            hf_decorrelate=True,
            refined_transient=True,
            adaptive_transient=True
        )

        assert result["success"] is True

    def test_output_format_preservation(self):
        """Test output format preservation"""
        input_file = self.create_test_audio_file("input_format.wav")

        result = preserving_sanitize(
            input_file,
            paranoid_mode=False,
            output_format="preserve"
        )

        assert result["success"] is True
        # Output should have same extension as input
        output_path = Path(result["output_file"])
        assert output_path.suffix == ".wav"

    def test_stats_structure(self):
        """Test stats structure in result"""
        input_file = self.create_test_audio_file("input_stats.wav")

        result = preserving_sanitize(
            input_file,
            paranoid_mode=False,
            threat_count=5
        )

        assert result["success"] is True
        assert "stats" in result
        assert "metadata_removed" in result["stats"]
        assert "watermarks_removed" in result["stats"]
        assert "processing_time" in result["stats"]
        assert "processing_speed" in result["stats"]

    def test_nonexistent_input_file(self):
        """Test with nonexistent input file"""
        result = preserving_sanitize(
            Path("/nonexistent/file.wav"),
            paranoid_mode=False
        )

        assert result["success"] is False
        assert "error" in result
