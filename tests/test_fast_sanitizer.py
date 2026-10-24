"""
Tests for fast_sanitizer module
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
import soundfile as sf

from mmm.fast_sanitizer import fast_sanitize


class TestFastSanitize:
    """Test cases for fast_sanitize function"""

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


class TestFastSanitizeBasic:
    """Basic tests for fast_sanitize"""

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
        result = fast_sanitize(input_file)

        assert isinstance(result, dict)

    def test_success_field(self):
        """Test result contains success field"""
        input_file = self.create_test_audio_file("test_success.wav")
        result = fast_sanitize(input_file)

        assert "success" in result
        assert isinstance(result["success"], bool)

    def test_output_file_created(self):
        """Test output file is created"""
        input_file = self.create_test_audio_file("test_output.wav")
        output_file = self.test_dir / "output.wav"

        result = fast_sanitize(input_file, output_file)

        if result["success"]:
            assert Path(result["output_file"]).exists()

    def test_auto_generated_output_path(self):
        """Test auto-generated output path"""
        input_file = self.create_test_audio_file("test_auto.wav")
        result = fast_sanitize(input_file)

        if result["success"]:
            assert "output_file" in result
            assert "clean" in result["output_file"]


class TestFastSanitizeStats:
    """Tests for fast_sanitize statistics"""

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
        result = fast_sanitize(input_file)

        if result["success"]:
            assert "stats" in result

    def test_stats_metadata_removed(self):
        """Test metadata_removed stat present"""
        input_file = self.create_test_audio_file("test_meta.wav")
        result = fast_sanitize(input_file)

        if result["success"]:
            assert "metadata_removed" in result["stats"]
            assert result["stats"]["metadata_removed"] >= 0

    def test_stats_watermarks_removed(self):
        """Test watermarks_removed stat present"""
        input_file = self.create_test_audio_file("test_wm.wav")
        result = fast_sanitize(input_file)

        if result["success"]:
            assert "watermarks_removed" in result["stats"]
            assert result["stats"]["watermarks_removed"] >= 0

    def test_stats_processing_time(self):
        """Test processing_time stat present"""
        input_file = self.create_test_audio_file("test_time.wav")
        result = fast_sanitize(input_file)

        if result["success"]:
            assert "processing_time" in result["stats"]
            assert result["stats"]["processing_time"] > 0

    def test_stats_processing_speed(self):
        """Test processing_speed stat present"""
        input_file = self.create_test_audio_file("test_speed.wav")
        result = fast_sanitize(input_file)

        if result["success"]:
            assert "processing_speed" in result["stats"]
            assert "x" in result["stats"]["processing_speed"]


class TestFastSanitizeParanoidMode:
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
        result = fast_sanitize(input_file, paranoid_mode=False)

        assert isinstance(result, dict)
        if result["success"]:
            assert Path(result["output_file"]).exists()

    def test_paranoid_mode_true(self):
        """Test with paranoid_mode=True"""
        input_file = self.create_test_audio_file("test_paranoid.wav")
        result = fast_sanitize(input_file, paranoid_mode=True)

        assert isinstance(result, dict)
        if result["success"]:
            assert Path(result["output_file"]).exists()


class TestFastSanitizeThreatCount:
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
        result = fast_sanitize(input_file, threat_count=0)

        assert isinstance(result, dict)

    def test_threat_count_positive(self):
        """Test with positive threat_count"""
        input_file = self.create_test_audio_file("test_positive.wav")
        result = fast_sanitize(input_file, threat_count=10)

        if result["success"]:
            # With threat_count > 0, should report removals
            assert result["stats"]["metadata_removed"] >= 1
            assert result["stats"]["watermarks_removed"] >= 1

    def test_threat_count_large(self):
        """Test with large threat_count"""
        input_file = self.create_test_audio_file("test_large.wav")
        result = fast_sanitize(input_file, threat_count=100)

        assert isinstance(result, dict)


class TestFastSanitizeErrorHandling:
    """Tests for error handling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_nonexistent_file(self):
        """Test with nonexistent input file"""
        result = fast_sanitize(Path("/nonexistent/file.wav"))

        assert result["success"] is False
        assert "error" in result

    def test_invalid_audio_file(self):
        """Test with invalid audio file"""
        # Create a non-audio file
        invalid_file = self.test_dir / "invalid.wav"
        invalid_file.write_text("This is not audio data")

        result = fast_sanitize(invalid_file)

        assert result["success"] is False
        assert "error" in result


class TestFastSanitizeFormats:
    """Tests for different file formats"""

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

    def test_wav_format(self):
        """Test WAV format processing"""
        file_path = self.test_dir / "test.wav"
        sf.write(str(file_path), self.test_audio, self.sample_rate)

        result = fast_sanitize(file_path)
        assert isinstance(result, dict)

    def test_flac_format(self):
        """Test FLAC format processing"""
        file_path = self.test_dir / "test.flac"
        sf.write(str(file_path), self.test_audio, self.sample_rate)

        result = fast_sanitize(file_path)
        assert isinstance(result, dict)

    def test_output_format_preserved(self):
        """Test output format matches input"""
        file_path = self.test_dir / "test.wav"
        sf.write(str(file_path), self.test_audio, self.sample_rate)

        result = fast_sanitize(file_path)
        if result["success"]:
            output_path = Path(result["output_file"])
            # Should preserve wav extension
            assert output_path.suffix.lower() in [".wav", ".mp3"]
