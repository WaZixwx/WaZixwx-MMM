#!/usr/bin/env python3
"""
Test GPU acceleration
"""

import sys
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
import numpy as np
import librosa
import time

def test_gpu_acceleration():
    print("🚀 Testing GPU Acceleration")
    print("=" * 50)

    # Test GPU availability
    print("\n1. Testing GPU packages...")
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")

    try:
        import cupy as cp
        print(f"✅ CuPy installed: {cp.__version__}")
        print(f"   CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    except ImportError as e:
        print(f"❌ CuPy not available: {e}")

    # Test optimized processor
    print("\n2. Testing Optimized Processor...")
    try:
        from mmm.optimized_processor import OptimizedAudioProcessor, GPUAcceleratedWatermarkDetector

        processor = OptimizedAudioProcessor(use_gpu=True, use_multiprocessing=True)

        # Load short audio sample for testing
        file_path = Path("Schizo Shaman.mp3")
        if file_path.exists():
            print(f"   Loading audio sample...")
            audio, sr = processor.load_audio_optimized(file_path)
            print(f"   Audio shape: {audio.shape}")
            print(f"   Sample rate: {sr}")
            print(f"   Duration: {len(audio)/sr:.1f} seconds")

            # Test GPU watermark detection on short sample
            short_audio = audio[:48000]  # 1 second
            print(f"   Testing GPU watermark detection on 1-second sample...")

            gpu_detector = GPUAcceleratedWatermarkDetector()
            start_time = time.time()
            result = gpu_detector.detect_spectral_patterns_gpu(short_audio, sr)
            elapsed = time.time() - start_time

            print(f"   ✅ GPU detection completed in {elapsed:.2f} seconds")
            print(f"   Result: {result}")

        else:
            print("❌ Schizo Shaman.mp3 not found")

    except Exception as e:
        print(f"❌ Optimized processor error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_acceleration()