"""
Tests for optimized_processor module
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

from mmm.optimized_processor import (
    OptimizedAudioProcessor,
    GPUAcceleratedWatermarkDetector,
    optimize_system,
)


class TestOptimizedAudioProcessor:
    """Test cases for OptimizedAudioProcessor class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 44100
        self.duration = 1.0

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


class TestOptimizedAudioProcessorInit:
    """Tests for OptimizedAudioProcessor initialization"""

    def test_default_initialization(self):
        """Test default initialization"""
        processor = OptimizedAudioProcessor()

        assert hasattr(processor, 'use_gpu')
        assert hasattr(processor, 'use_multiprocessing')
        assert hasattr(processor, 'num_cores')

    def test_gpu_disabled(self):
        """Test with GPU disabled"""
        processor = OptimizedAudioProcessor(use_gpu=False)

        assert processor.use_gpu is False

    def test_multiprocessing_disabled(self):
        """Test with multiprocessing disabled"""
        processor = OptimizedAudioProcessor(use_multiprocessing=False)

        assert processor.use_multiprocessing is False

    def test_num_cores_set(self):
        """Test num_cores is set"""
        processor = OptimizedAudioProcessor()

        assert processor.num_cores > 0


class TestLoadAudioOptimized:
    """Tests for load_audio_optimized method"""

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

    def test_loads_audio(self):
        """Test audio loading"""
        processor = OptimizedAudioProcessor(use_gpu=False)
        input_file = self.create_test_audio_file("test.wav")

        audio, sr = processor.load_audio_optimized(input_file)

        assert isinstance(audio, np.ndarray)
        assert sr == self.sample_rate

    def test_audio_shape(self):
        """Test loaded audio has correct shape"""
        processor = OptimizedAudioProcessor(use_gpu=False)
        input_file = self.create_test_audio_file("test_shape.wav")

        audio, sr = processor.load_audio_optimized(input_file)

        assert len(audio) > 0

    def test_custom_sample_rate(self):
        """Test loading with custom sample rate"""
        processor = OptimizedAudioProcessor(use_gpu=False)
        input_file = self.create_test_audio_file("test_sr.wav")

        audio, sr = processor.load_audio_optimized(input_file, sample_rate=22050)

        assert sr == 22050


class TestProcessInChunks:
    """Tests for process_in_chunks method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        self.duration = 2.0

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_processes_chunks(self):
        """Test chunk processing"""
        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)

        def simple_process(chunk, sample_rate):
            return {"processed": True, "length": len(chunk)}

        results = processor.process_in_chunks(
            self.test_audio,
            self.sample_rate,
            chunk_duration=0.5,
            chunk_overlap=0.1,
            process_func=simple_process
        )

        assert len(results) > 0
        assert all("processed" in r for r in results)

    def test_chunk_count(self):
        """Test correct number of chunks created"""
        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)

        def simple_process(chunk, sample_rate):
            return {"length": len(chunk)}

        results = processor.process_in_chunks(
            self.test_audio,
            self.sample_rate,
            chunk_duration=1.0,
            chunk_overlap=0.0,
            process_func=simple_process
        )

        # With 2 second audio and 1 second chunks, should have at least 2 chunks
        assert len(results) >= 2


class TestDetectWatermarksParallel:
    """Tests for detect_watermarks_parallel method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        self.duration = 2.0

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_returns_dict(self):
        """Test returns dictionary"""
        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)

        result = processor.detect_watermarks_parallel(
            self.test_audio,
            self.sample_rate,
            chunk_duration=1.0
        )

        assert isinstance(result, dict)

    def test_result_structure(self):
        """Test result has expected structure"""
        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)

        result = processor.detect_watermarks_parallel(
            self.test_audio,
            self.sample_rate,
            chunk_duration=1.0
        )

        assert "detected" in result
        assert "method_results" in result
        assert "confidence_scores" in result
        assert "watermark_count" in result
        assert "chunk_count" in result

    def test_chunk_count_recorded(self):
        """Test chunk count is recorded"""
        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)

        result = processor.detect_watermarks_parallel(
            self.test_audio,
            self.sample_rate,
            chunk_duration=1.0
        )

        assert result["chunk_count"] > 0


class TestGpuAcceleratedStft:
    """Tests for gpu_accelerated_stft method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        self.duration = 0.5

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_cpu_fallback(self):
        """Test CPU fallback when GPU disabled"""
        processor = OptimizedAudioProcessor(use_gpu=False)

        result = processor.gpu_accelerated_stft(self.test_audio)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_custom_parameters(self):
        """Test with custom n_fft and hop_length"""
        processor = OptimizedAudioProcessor(use_gpu=False)

        result = processor.gpu_accelerated_stft(
            self.test_audio,
            n_fft=1024,
            hop_length=256
        )

        assert isinstance(result, np.ndarray)


class TestGPUAcceleratedWatermarkDetector:
    """Test cases for GPUAcceleratedWatermarkDetector class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        self.duration = 0.5

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)


class TestGPUAcceleratedWatermarkDetectorInit:
    """Tests for GPUAcceleratedWatermarkDetector initialization"""

    def test_initialization(self):
        """Test basic initialization"""
        detector = GPUAcceleratedWatermarkDetector()

        assert hasattr(detector, 'gpu_available')
        assert hasattr(detector, 'device')


class TestDetectSpectralPatternsGpu:
    """Tests for detect_spectral_patterns_gpu method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_rate = 44100
        self.duration = 0.5

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_returns_dict(self):
        """Test returns dictionary"""
        detector = GPUAcceleratedWatermarkDetector()

        result = detector.detect_spectral_patterns_gpu(self.test_audio, self.sample_rate)

        assert isinstance(result, dict)

    def test_result_structure(self):
        """Test result has expected structure"""
        detector = GPUAcceleratedWatermarkDetector()

        result = detector.detect_spectral_patterns_gpu(self.test_audio, self.sample_rate)

        assert "detected" in result
        assert "confidence" in result

    def test_confidence_range(self):
        """Test confidence is a reasonable number"""
        detector = GPUAcceleratedWatermarkDetector()

        result = detector.detect_spectral_patterns_gpu(self.test_audio, self.sample_rate)

        assert isinstance(result["confidence"], (int, float))
        assert result["confidence"] >= 0


class TestBatchProcessFiles:
    """Tests for batch_process_files method"""

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

    def test_processes_single_file(self):
        """Test processing single file"""
        detector = GPUAcceleratedWatermarkDetector()
        input_file = self.create_test_audio_file("test.wav")

        results = detector.batch_process_files([input_file], chunk_duration=1.0)

        assert isinstance(results, list)
        assert len(results) == 1

    def test_processes_multiple_files(self):
        """Test processing multiple files"""
        detector = GPUAcceleratedWatermarkDetector()

        files = [
            self.create_test_audio_file("test1.wav"),
            self.create_test_audio_file("test2.wav")
        ]

        results = detector.batch_process_files(files, chunk_duration=1.0)

        assert isinstance(results, list)
        assert len(results) == 2

    def test_empty_file_list(self):
        """Test with empty file list"""
        detector = GPUAcceleratedWatermarkDetector()

        results = detector.batch_process_files([], chunk_duration=1.0)

        assert isinstance(results, list)
        assert len(results) == 0


class TestOptimizeSystem:
    """Tests for optimize_system function"""

    def test_sets_environment_variables(self):
        """Test environment variables are set"""
        import os

        # Call optimize_system
        optimize_system()

        # Check environment variables are set
        assert "OMP_NUM_THREADS" in os.environ
        assert "MKL_NUM_THREADS" in os.environ
        assert "NUMBA_NUM_THREADS" in os.environ

    def test_environment_values_positive(self):
        """Test environment values are positive integers"""
        import os

        optimize_system()

        assert int(os.environ["OMP_NUM_THREADS"]) > 0
        assert int(os.environ["MKL_NUM_THREADS"]) > 0
        assert int(os.environ["NUMBA_NUM_THREADS"]) > 0


class TestOptimizedAudioProcessorEdgeCases:
    """Edge case tests"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_rate = 44100

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_very_short_audio(self):
        """Test with very short audio"""
        t = np.linspace(0, 0.1, int(self.sample_rate * 0.1))  # 100ms
        short_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        file_path = self.test_dir / "short.wav"
        sf.write(str(file_path), short_audio, self.sample_rate)

        processor = OptimizedAudioProcessor(use_gpu=False)
        audio, sr = processor.load_audio_optimized(file_path)

        assert len(audio) > 0

    def test_silent_audio(self):
        """Test with silent audio"""
        silent = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence

        file_path = self.test_dir / "silent.wav"
        sf.write(str(file_path), silent, self.sample_rate)

        processor = OptimizedAudioProcessor(use_gpu=False)
        audio, sr = processor.load_audio_optimized(file_path)

        assert len(audio) > 0
        assert np.max(np.abs(audio)) == 0  # Should still be silent

    def test_stereo_audio(self):
        """Test with stereo audio"""
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5))
        stereo = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 880 * t)
        ]).astype(np.float32)

        file_path = self.test_dir / "stereo.wav"
        sf.write(str(file_path), stereo, self.sample_rate)

        processor = OptimizedAudioProcessor(use_gpu=False)
        audio, sr = processor.load_audio_optimized(file_path)

        # Should load successfully (may convert to mono)
        assert len(audio) > 0


class TestOptimizedAudioProcessorGuards:
    """Tests for input guardrails in optimized processor methods."""

    def test_process_in_chunks_rejects_invalid_overlap(self):
        """chunk_overlap must be smaller than chunk_duration."""
        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)
        audio = np.zeros(44100, dtype=np.float32)

        with pytest.raises(ValueError, match="chunk_overlap"):
            processor.process_in_chunks(
                audio,
                44100,
                chunk_duration=1.0,
                chunk_overlap=1.0,
                process_func=lambda chunk, sample_rate: len(chunk),
            )

    def test_detect_watermarks_parallel_uses_picklable_worker(self, monkeypatch):
        """Multiprocessing should use a top-level picklable worker function."""
        qualnames = []

        class FakeExecutor:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, iterable):
                qualnames.append(getattr(func, "__qualname__", ""))
                items = list(iterable)
                return [
                    {
                        "detected": [],
                        "method_results": {},
                        "confidence_scores": {},
                        "watermark_count": 0,
                        "overall_confidence": 0.0,
                    }
                    for _ in items
                ]

        monkeypatch.setattr("mmm.optimized_processor.ProcessPoolExecutor", FakeExecutor)

        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=True)
        audio = np.random.randn(88200).astype(np.float32)
        result = processor.detect_watermarks_parallel(audio, 44100, chunk_duration=1.0)

        assert qualnames
        assert "<locals>" not in qualnames[0]
        assert result["chunk_count"] == 2

    def test_optimize_librosa_performance_no_legacy_api_crash(self):
        """Optimization helper should not rely on removed librosa APIs."""
        processor = OptimizedAudioProcessor(use_gpu=False, use_multiprocessing=False)
        processor.optimize_librosa_performance()
