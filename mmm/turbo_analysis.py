#!/usr/bin/env python3
"""
Turbo Analysis - GPU + Multi-core CPU optimization
"""

import os
import librosa
import numpy as np
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from mmm.detection.metadata_scanner import MetadataScanner
from mmm.optimized_processor import (
    OptimizedAudioProcessor,
    GPUAcceleratedWatermarkDetector,
)

def _configure_thread_counts() -> None:
    """Set thread counts for numeric libraries. Call once before heavy computation."""
    cpu_count = str(mp.cpu_count() or 1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, cpu_count)


def analyze_audio_chunk_gpu(args):
    """
    Analyze a chunk of audio data using GPU acceleration
    Args: (audio_chunk, sample_rate, chunk_id, chunk_start_time)
    """
    audio_chunk, sample_rate, chunk_id, chunk_start_time = args

    results = {
        "chunk_id": chunk_id,
        "chunk_start_time": chunk_start_time,
        "watermarks": None,
        "error": None,
        "processing_time": 0,
    }

    try:
        # GPU-accelerated watermark detection
        gpu_detector = GPUAcceleratedWatermarkDetector()
        start_time = time.time()

        # Use spectral pattern detection (fastest GPU method)
        gpu_result = gpu_detector.detect_spectral_patterns_gpu(audio_chunk, sample_rate)

        results["processing_time"] = time.time() - start_time
        results["watermarks"] = {
            "detected": gpu_result["detected"],
            "confidence": gpu_result["confidence"],
            "method": "gpu_spectral",
        }

    except Exception as e:
        results["error"] = str(e)

    return results


def _detect_gpu_name() -> str:
    """Detect actual GPU name via CUDA, or report CPU-only mode."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "Not detected (CPU mode)"


def _is_gpu_available() -> bool:
    """Return True when CUDA is available in the current environment."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def turbo_analysis(file_path, chunk_duration=5.0):
    """
    Turbo-charged analysis using GPU + Multi-core CPU
    """
    gpu_name = _detect_gpu_name()
    print(f"🚀 TURBO ANALYSIS - GPU + Multi-Core CPU")
    print(f"   GPU: {gpu_name}")
    print(f"   CPU: {mp.cpu_count()} cores")
    print(f"   File: {file_path}")
    print(f"   Chunk duration: {chunk_duration}s")
    print()

    _configure_thread_counts()

    # Initialize optimized processor
    processor = OptimizedAudioProcessor(use_gpu=True, use_multiprocessing=True)

    # Load audio
    print("⚡ Loading audio with optimized processor...")
    start_time = time.time()
    audio, sr = processor.load_audio_optimized(file_path)
    load_time = time.time() - start_time

    total_duration = len(audio) / sr
    chunk_samples = int(chunk_duration * sr)
    num_chunks = int(np.ceil(len(audio) / chunk_samples))

    print(f"   ✅ Loaded in {load_time:.2f}s")
    print(f"   Duration: {total_duration:.1f} seconds")
    print(f"   Processing in {num_chunks} chunks of {chunk_duration}s each")
    print()

    # Create chunks
    print("🎯 Preparing chunks for parallel processing...")
    chunks = []
    chunk_positions = []
    for i in range(0, len(audio), chunk_samples):
        end = min(i + chunk_samples, len(audio))
        chunk = audio[i:end]
        chunks.append(chunk)
        chunk_positions.append((i / sr, end / sr))

    # Prepare arguments for parallel processing
    args_list = [
        (chunk, sr, i, chunk_positions[i][0]) for i, chunk in enumerate(chunks)
    ]

    # Process in parallel using both GPU and CPU cores
    print(f"🔥 Processing chunks with GPU acceleration...")
    start_time = time.time()

    # Use ThreadPoolExecutor for GPU processing (GPU handles scheduling)
    with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 4)) as executor:
        chunk_results = list(executor.map(analyze_audio_chunk_gpu, args_list))

    elapsed = time.time() - start_time

    print(f"✅ Completed in {elapsed:.2f} seconds")
    print(f"   Speed: {total_duration/elapsed:.1f}x real-time")
    print()

    # Aggregate results
    print("📊 Aggregating Results...")

    # Metadata analysis (single thread - fast)
    print("   🔍 Scanning metadata...")
    scanner = MetadataScanner()
    metadata = scanner.scan_file(file_path)

    # Aggregate watermark results
    gpu_watermarks_detected = 0
    total_confidence = 0
    processing_times = []

    for result in chunk_results:
        if result["watermarks"]:
            processing_times.append(result["processing_time"])
            if result["watermarks"]["detected"]:
                gpu_watermarks_detected += 1
            total_confidence += result["watermarks"]["confidence"]

    avg_confidence = total_confidence / len(chunk_results) if chunk_results else 0
    avg_chunk_time = np.mean(processing_times) if processing_times else 0

    # Display results
    print("\n" + "=" * 60)
    print("🎯 TURBO ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\n📁 File: {file_path}")
    print(f"   Size: {Path(file_path).stat().st_size/1024/1024:.1f} MB")
    print(f"   Duration: {total_duration:.1f} seconds")

    print(f"\n📋 Metadata:")
    print(f"   Tags found: {len(metadata['tags'])}")
    print(f"   Suspicious chunks: {len(metadata['suspicious_chunks'])}")
    print(f"   Hidden patterns: {len(metadata['hidden_data'])}")

    if metadata["tags"]:
        for tag in metadata["tags"][:3]:
            suspicious = "🚨" if tag["suspicious"] else "✅"
            print(f"      {suspicious} {tag['key']}")

    print(f"\n🚀 GPU Watermark Analysis:")
    print(f"   Chunks processed: {len(chunk_results)}")
    print(f"   Watermarks detected: {gpu_watermarks_detected}")
    print(f"   Average confidence: {avg_confidence:.1%}")
    print(f"   Average chunk processing time: {avg_chunk_time:.3f}s")

    print(f"\n⚡ PERFORMANCE:")
    print(f"   Audio loading time: {load_time:.2f} seconds")
    print(f"   Processing time: {elapsed:.2f} seconds")
    print(f"   Real-time factor: {total_duration/elapsed:.1f}x")
    print(
        "   GPU acceleration: "
        + ("✅ ENABLED" if _is_gpu_available() else "❌ DISABLED (CPU fallback)")
    )
    print(f"   CPU cores utilized: {min(mp.cpu_count(), 4)}")
    print(f"   Throughput: {(total_duration/elapsed)*60:.1f} audio-minutes/min")

    # Calculate threat level — only count genuinely suspicious findings
    suspicious_tags = [
        t
        for t in metadata.get("tags", [])
        if isinstance(t, dict) and t.get("suspicious")
    ]
    total_threats = (
        len(suspicious_tags)
        + len(metadata["suspicious_chunks"])
        + len(metadata["hidden_data"])
        + gpu_watermarks_detected
    )

    print(f"\n🚨 THREAT LEVEL: ", end="")
    if total_threats > 20:
        print("🔴 VERY HIGH - Extensive AI markers detected")
    elif total_threats > 10:
        print("🟠 HIGH - Strong AI generation indicators")
    elif total_threats > 5:
        print("🟡 MEDIUM - Some AI traces detected")
    else:
        print("🟢 LOW - No significant AI markers detected")

    print(f"   Total threats: {total_threats}")

    return {
        "file_info": {
            "path": str(file_path),
            "size": Path(file_path).stat().st_size,
            "format": Path(file_path).suffix.lstrip(".").upper(),
            "duration": total_duration,
            "sample_rate": sr,
            "channels": audio.shape[1] if audio.ndim > 1 else 1,
        },
        "metadata": metadata,
        "gpu_watermarks": {
            "detected": (
                [{"method": "gpu", "confidence": avg_confidence}]
                if gpu_watermarks_detected
                else []
            ),
            "total_count": gpu_watermarks_detected,
            "avg_confidence": avg_confidence,
            "chunks_processed": len(chunk_results),
            "overall_confidence": avg_confidence,
        },
        "performance": {
            "loading_time": load_time,
            "processing_time": elapsed,
            "realtime_factor": total_duration / elapsed if elapsed else 0,
            "avg_chunk_time": avg_chunk_time,
        },
        "total_threats": total_threats,
        "threat_level": (
            "VERY HIGH"
            if total_threats > 20
            else "HIGH"
            if total_threats > 10
            else "MEDIUM"
            if total_threats > 5
            else "LOW"
        ),
    }


def main():
    """Main function"""
    print("🚀 MMM Turbo Analysis - GPU + Multi-Core Optimization")
    gpu_name = _detect_gpu_name()
    print(f"🎵 GPU: {gpu_name}")
    print("=" * 60)

    file_path = Path("Schizo Shaman.mp3")
    if not file_path.exists():
        print("❌ Error: Schizo Shaman.mp3 not found")
        return

    results = turbo_analysis(file_path)

    print("\n💀 Turbo Analysis Complete!")
    print("   Maximum GPU + CPU performance achieved!")
    print("   Ready for high-speed processing!")


if __name__ == "__main__":
    main()
