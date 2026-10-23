"""
Tests for turbo_analysis module
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
from unittest.mock import patch, MagicMock

from mmm.turbo_analysis import analyze_audio_chunk_gpu, turbo_analysis


class TestAnalyzeAudioChunkGpu:
    """Test cases for analyze_audio_chunk_gpu function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_returns_result_dict(self):
        """Test that function returns a result dictionary"""
        args = (self.test_audio, self.sample_rate, 0, 0.0)
        result = analyze_audio_chunk_gpu(args)

        assert isinstance(result, dict)
        assert "chunk_id" in result
        assert "chunk_start_time" in result
        assert "watermarks" in result
        assert "error" in result
        assert "processing_time" in result

    def test_chunk_id_preserved(self):
        """Test chunk_id is preserved in result"""
        args = (self.test_audio, self.sample_rate, 42, 5.0)
        result = analyze_audio_chunk_gpu(args)

        assert result["chunk_id"] == 42
        assert result["chunk_start_time"] == 5.0

    def test_processing_time_recorded(self):
        """Test processing time is recorded"""
        args = (self.test_audio, self.sample_rate, 0, 0.0)
        result = analyze_audio_chunk_gpu(args)

        assert result["processing_time"] >= 0

    def test_handles_empty_audio(self):
        """Test handling of empty audio array"""
        empty_audio = np.array([], dtype=np.float32)
        args = (empty_audio, self.sample_rate, 0, 0.0)

        # Should handle gracefully (either return error or empty result)
        result = analyze_audio_chunk_gpu(args)
        assert isinstance(result, dict)

    def test_watermarks_structure(self):
        """Test watermarks field structure when not None"""
        args = (self.test_audio, self.sample_rate, 0, 0.0)
        result = analyze_audio_chunk_gpu(args)

        wm = result["watermarks"]
        if wm is not None:
            assert isinstance(wm, dict)
            assert "detected" in wm
            assert "confidence" in wm
            assert "method" in wm


class TestTurboAnalysis:
    """Test cases for turbo_analysis function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 44100
        self.duration = 2.0  # 2 seconds for chunking tests

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

    def test_returns_result_dict(self):
        """Test that function returns a result dictionary"""
        input_file = self.create_test_audio_file("test_turbo.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert isinstance(result, dict)

    def test_result_contains_file_info(self):
        """Test result contains file info"""
        input_file = self.create_test_audio_file("test_file_info.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert "file_info" in result
        assert "path" in result["file_info"]
        assert "size" in result["file_info"]
        assert "duration" in result["file_info"]
        assert "sample_rate" in result["file_info"]

    def test_result_contains_metadata(self):
        """Test result contains metadata analysis"""
        input_file = self.create_test_audio_file("test_metadata.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert "metadata" in result

    def test_result_contains_gpu_watermarks(self):
        """Test result contains GPU watermark analysis"""
        input_file = self.create_test_audio_file("test_gpu_wm.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert "gpu_watermarks" in result
        assert "detected" in result["gpu_watermarks"]
        assert "total_count" in result["gpu_watermarks"]
        assert "avg_confidence" in result["gpu_watermarks"]
        assert "chunks_processed" in result["gpu_watermarks"]

    def test_result_contains_performance(self):
        """Test result contains performance metrics"""
        input_file = self.create_test_audio_file("test_perf.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert "performance" in result
        assert "loading_time" in result["performance"]
        assert "processing_time" in result["performance"]
        assert "realtime_factor" in result["performance"]

    def test_result_contains_threat_level(self):
        """Test result contains threat level"""
        input_file = self.create_test_audio_file("test_threat.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert "total_threats" in result
        assert "threat_level" in result
        assert result["threat_level"] in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]

    def test_chunk_duration_affects_processing(self):
        """Test different chunk durations"""
        input_file = self.create_test_audio_file("test_chunk.wav")

        result_short = turbo_analysis(input_file, chunk_duration=0.5)
        result_long = turbo_analysis(input_file, chunk_duration=2.0)

        # Shorter chunks should result in more chunks processed
        assert result_short["gpu_watermarks"]["chunks_processed"] >= result_long["gpu_watermarks"]["chunks_processed"]

    def test_correct_file_duration(self):
        """Test file duration is correctly reported"""
        input_file = self.create_test_audio_file("test_duration.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        reported_duration = result["file_info"]["duration"]
        assert abs(reported_duration - self.duration) < 0.1  # Within 100ms

    def test_correct_sample_rate(self):
        """Test sample rate is correctly reported"""
        input_file = self.create_test_audio_file("test_sr.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert result["file_info"]["sample_rate"] == self.sample_rate

    def test_file_size_reported(self):
        """Test file size is reported"""
        input_file = self.create_test_audio_file("test_size.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert result["file_info"]["size"] > 0

    def test_processing_time_positive(self):
        """Test processing time is positive"""
        input_file = self.create_test_audio_file("test_time.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert result["performance"]["processing_time"] > 0

    def test_realtime_factor_calculated(self):
        """Test realtime factor is calculated"""
        input_file = self.create_test_audio_file("test_realtime.wav")
        result = turbo_analysis(input_file, chunk_duration=1.0)

        assert result["performance"]["realtime_factor"] > 0


class TestTurboAnalysisEdgeCases:
    """Edge case tests for turbo_analysis"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 44100

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str, duration: float) -> Path:
        """Create a test audio file"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        file_path = self.test_dir / filename
        sf.write(str(file_path), audio, self.sample_rate)
        return file_path

    def test_very_short_audio(self):
        """Test with very short audio (< 1 chunk)"""
        input_file = self.create_test_audio_file("short.wav", 0.5)
        result = turbo_analysis(input_file, chunk_duration=5.0)

        assert isinstance(result, dict)
        assert result["gpu_watermarks"]["chunks_processed"] >= 1

    def test_chunk_larger_than_file(self):
        """Test when chunk duration exceeds file duration"""
        input_file = self.create_test_audio_file("small.wav", 1.0)
        result = turbo_analysis(input_file, chunk_duration=10.0)

        assert isinstance(result, dict)
        assert result["gpu_watermarks"]["chunks_processed"] == 1

    def test_stereo_audio(self):
        """Test with stereo audio"""
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        stereo = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 880 * t)
        ]).astype(np.float32)

        file_path = self.test_dir / "stereo.wav"
        sf.write(str(file_path), stereo, self.sample_rate)

        result = turbo_analysis(file_path, chunk_duration=0.5)
        assert isinstance(result, dict)

    def test_silent_audio(self):
        """Test with silent audio"""
        duration = 1.0
        silent = np.zeros(int(self.sample_rate * duration), dtype=np.float32)

        file_path = self.test_dir / "silent.wav"
        sf.write(str(file_path), silent, self.sample_rate)

        result = turbo_analysis(file_path, chunk_duration=0.5)
        assert isinstance(result, dict)
