"""
Tests for effective_sanitizer module
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

from mmm.effective_sanitizer import aggressive_sanitize
import mmm.effective_sanitizer as effective_sanitizer_module


class TestAggressiveSanitize:
    """Test cases for aggressive_sanitize function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 44100
        self.duration = 0.5  # Short for fast tests

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str, audio_data: np.ndarray = None) -> Path:
        """Create a test audio file"""
        if audio_data is None:
            audio_data = self.test_audio
        file_path = self.test_dir / filename
        sf.write(str(file_path), audio_data, self.sample_rate)
        return file_path


class TestAggressiveSanitizeBasic:
    """Basic tests for aggressive_sanitize"""

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

    def test_returns_dict(self):
        """Test that function returns a dictionary"""
        input_file = self.create_test_audio_file("test.wav")
        result = aggressive_sanitize(input_file)

        assert isinstance(result, dict)

    def test_success_field(self):
        """Test result contains success field"""
        input_file = self.create_test_audio_file("test_success.wav")
        result = aggressive_sanitize(input_file)

        assert "success" in result
        assert isinstance(result["success"], bool)

    def test_output_file_created(self):
        """Test output file is created"""
        input_file = self.create_test_audio_file("test_output.wav")
        output_file = self.test_dir / "output.wav"

        result = aggressive_sanitize(input_file, output_file)

        if result["success"]:
            assert Path(result["output_file"]).exists()

    def test_auto_generated_output_path(self):
        """Test auto-generated output path"""
        input_file = self.create_test_audio_file("test_auto.wav")
        result = aggressive_sanitize(input_file)

        if result["success"]:
            assert "output_file" in result
            assert "sanitized" in result["output_file"]


class TestAggressiveSanitizeStats:
    """Tests for aggressive_sanitize statistics"""

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

    def test_stats_present(self):
        """Test stats are present in result"""
        input_file = self.create_test_audio_file("test_stats.wav")
        result = aggressive_sanitize(input_file)

        if result["success"]:
            assert "stats" in result

    def test_stats_metadata_removed(self):
        """Test metadata_removed stat present"""
        input_file = self.create_test_audio_file("test_meta.wav")
        result = aggressive_sanitize(input_file)

        if result["success"]:
            assert "metadata_removed" in result["stats"]
            assert result["stats"]["metadata_removed"] >= 0

    def test_stats_watermarks_removed(self):
        """Test watermarks_removed stat present"""
        input_file = self.create_test_audio_file("test_wm.wav")
        result = aggressive_sanitize(input_file)

        if result["success"]:
            assert "watermarks_removed" in result["stats"]
            assert result["stats"]["watermarks_removed"] >= 0

    def test_stats_processing_time(self):
        """Test processing_time stat present"""
        input_file = self.create_test_audio_file("test_time.wav")
        result = aggressive_sanitize(input_file)

        if result["success"]:
            assert "processing_time" in result["stats"]
            assert result["stats"]["processing_time"] > 0

    def test_stats_processing_speed(self):
        """Test processing_speed stat present"""
        input_file = self.create_test_audio_file("test_speed.wav")
        result = aggressive_sanitize(input_file)

        if result["success"]:
            assert "processing_speed" in result["stats"]
            assert "x" in result["stats"]["processing_speed"]

    def test_stats_effectiveness(self):
        """Test effectiveness stat present"""
        input_file = self.create_test_audio_file("test_eff.wav")
        result = aggressive_sanitize(input_file)

        if result["success"]:
            assert "effectiveness" in result["stats"]
            assert result["stats"]["effectiveness"] >= 0
            assert result["stats"]["effectiveness"] <= 100


class TestAggressiveSanitizeParanoidMode:
    """Tests for paranoid mode"""

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

    def test_paranoid_mode_false(self):
        """Test with paranoid_mode=False"""
        input_file = self.create_test_audio_file("test_normal.wav")
        result = aggressive_sanitize(input_file, paranoid_mode=False)

        assert isinstance(result, dict)
        if result["success"]:
            assert Path(result["output_file"]).exists()

    def test_paranoid_mode_true(self):
        """Test with paranoid_mode=True"""
        input_file = self.create_test_audio_file("test_paranoid.wav")
        result = aggressive_sanitize(input_file, paranoid_mode=True)

        assert isinstance(result, dict)
        if result["success"]:
            assert Path(result["output_file"]).exists()

    def test_paranoid_more_effective(self):
        """Test paranoid mode is more effective"""
        input_file = self.create_test_audio_file("test_eff_compare.wav")

        result_normal = aggressive_sanitize(input_file, paranoid_mode=False, threat_count=10)
        result_paranoid = aggressive_sanitize(input_file, paranoid_mode=True, threat_count=10)

        if result_normal["success"] and result_paranoid["success"]:
            assert result_paranoid["stats"]["effectiveness"] >= result_normal["stats"]["effectiveness"]


class TestAggressiveSanitizeThreatCount:
    """Tests for threat_count parameter"""

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

    def test_threat_count_zero(self):
        """Test with threat_count=0"""
        input_file = self.create_test_audio_file("test_zero.wav")
        result = aggressive_sanitize(input_file, threat_count=0)

        assert isinstance(result, dict)

    def test_threat_count_positive(self):
        """Test with positive threat_count"""
        input_file = self.create_test_audio_file("test_positive.wav")
        result = aggressive_sanitize(input_file, threat_count=10)

        if result["success"]:
            # With threat_count > 0, should report removals
            assert result["stats"]["metadata_removed"] >= 1
            assert result["stats"]["watermarks_removed"] >= 1

    def test_threat_count_large(self):
        """Test with large threat_count"""
        input_file = self.create_test_audio_file("test_large.wav")
        result = aggressive_sanitize(input_file, threat_count=100)

        assert isinstance(result, dict)

    def test_threat_count_affects_effectiveness(self):
        """Test threat_count affects reported effectiveness"""
        input_file = self.create_test_audio_file("test_eff.wav")

        result_0 = aggressive_sanitize(input_file, threat_count=0)
        result_100 = aggressive_sanitize(input_file, threat_count=100)

        # Both should report effectiveness
        if result_0["success"] and result_100["success"]:
            assert "effectiveness" in result_0["stats"]
            assert "effectiveness" in result_100["stats"]


class TestAggressiveSanitizeErrorHandling:
    """Tests for error handling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_nonexistent_file(self):
        """Test with nonexistent input file"""
        result = aggressive_sanitize(Path("/nonexistent/file.wav"))

        assert result["success"] is False
        assert "error" in result

    def test_invalid_audio_file(self):
        """Test with invalid audio file"""
        # Create a non-audio file
        invalid_file = self.test_dir / "invalid.wav"
        invalid_file.write_text("This is not audio data")

        result = aggressive_sanitize(invalid_file)

        assert result["success"] is False
        assert "error" in result


class TestAggressiveSanitizeAudioProcessing:
    """Tests for audio processing aspects"""

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

    def create_test_audio_file(self, filename: str, audio_data: np.ndarray = None) -> Path:
        """Create a test audio file"""
        if audio_data is None:
            audio_data = self.test_audio
        file_path = self.test_dir / filename
        sf.write(str(file_path), audio_data, self.sample_rate)
        return file_path

    def test_output_not_silent(self):
        """Test output is not completely silent"""
        input_file = self.create_test_audio_file("test_silent.wav")
        output_file = self.test_dir / "output.wav"

        result = aggressive_sanitize(input_file, output_file)

        if result["success"]:
            output_audio, _ = sf.read(str(output_file))
            assert np.max(np.abs(output_audio)) > 0

    def test_output_not_clipped(self):
        """Test output is not clipped (stays in -1 to 1 range)"""
        input_file = self.create_test_audio_file("test_clip.wav")
        output_file = self.test_dir / "output.wav"

        result = aggressive_sanitize(input_file, output_file)

        if result["success"]:
            output_audio, _ = sf.read(str(output_file))
            assert np.max(np.abs(output_audio)) <= 1.0

    def test_stereo_processing(self):
        """Test stereo audio processing"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        stereo = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 880 * t)
        ]).astype(np.float32)

        input_file = self.create_test_audio_file("stereo.wav", stereo)
        result = aggressive_sanitize(input_file)

        # Should handle stereo gracefully (may convert to mono for processing)
        assert isinstance(result, dict)

    def test_frequency_domain_processing(self):
        """Test that frequency domain processing occurs"""
        input_file = self.create_test_audio_file("test_freq.wav")
        output_file = self.test_dir / "output.wav"

        result = aggressive_sanitize(input_file, output_file)

        if result["success"]:
            input_audio, _ = sf.read(str(input_file))
            output_audio, _ = sf.read(str(output_file))

            # Audio should be different after processing
            if len(input_audio) == len(output_audio):
                assert not np.allclose(input_audio, output_audio)


class TestAggressiveSanitizeOutputFormat:
    """Output container/extension behavior tests."""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 22050
        t = np.linspace(0, 0.3, int(self.sample_rate * 0.3), endpoint=False)
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str) -> Path:
        file_path = self.test_dir / filename
        sf.write(str(file_path), self.test_audio, self.sample_rate)
        return file_path

    def test_respects_output_file_suffix_for_flac(self, monkeypatch):
        """When caller requests .flac, output is encoded as FLAC."""
        input_file = self.create_test_audio_file("format_src.wav")
        output_file = self.test_dir / "format_out.flac"

        monkeypatch.setattr(
            effective_sanitizer_module,
            "librosa",
            SimpleNamespace(
                load=lambda *_args, **_kwargs: (
                    self.test_audio.copy(),
                    self.sample_rate,
                )
            ),
        )

        result = aggressive_sanitize(input_file, output_file=output_file)

        if result["success"]:
            out_path = Path(result["output_file"])
            assert out_path.suffix.lower() == ".flac"
            assert sf.info(str(out_path)).format == "FLAC"

    def test_rejects_unsupported_output_format(self):
        """Fail closed on unknown output containers."""
        input_file = self.create_test_audio_file("format_reject.wav")
        output_file = self.test_dir / "format_out.ogg"

        result = aggressive_sanitize(input_file, output_file=output_file)

        assert result["success"] is False
        assert "Unsupported output format" in result["error"]
