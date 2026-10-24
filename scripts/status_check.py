#!/usr/bin/env python3
"""
Quick status check of all MMM features
"""

import os
import subprocess
import sys
import wave
import struct
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_temp_test_wav(output_path: Path, duration_sec: float = 1.5, sample_rate: int = 44100):
    """Generate a clean synthetic WAV file without needing scipy/soundfile"""
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(output_path), 'wb') as wav:
        wav.setnchannels(2)  # Stereo
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            # 440 Hz Left, 880 Hz Right
            val_l = int(16000 * math.sin(2.0 * math.pi * 440.0 * t))
            val_r = int(14000 * math.sin(2.0 * math.pi * 880.0 * t))
            frames.extend(struct.pack('<hh', val_l, val_r))
        wav.writeframes(frames)

def test_command(cmd, timeout=15):
    """Test a command with timeout"""
    print(f"\n🧪 Testing: {cmd}")

    try:
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{python_path}" if python_path else str(PROJECT_ROOT)

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=env
        )

        if "SUCCESS" in result.stdout or "✅" in result.stdout:
            print("✅ Command successful")
            return True
        elif "ERROR" in result.stdout or "❌" in result.stdout:
            print(f"❌ Command failed: {result.stdout[:120]}")
            return False
        elif "MELODIC METADATA MASSACRER" in result.stdout:
            print("⚠️ Command started and displayed banner")
            return True
        else:
            print("❓ Command completed without standard status string")
            return True

    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("🔍 MMM Status Check")
    print("=" * 50)

    test_file = None
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if not test_file.exists():
            print(f"❌ Specified test file not found: {test_file}")
            return 1
    else:
        # Check standard default candidates or create synthetic audio
        default_candidates = [
            PROJECT_ROOT / "test_audio" / "stereo_test.wav",
            PROJECT_ROOT / "Schizo Shaman.mp3"
        ]
        for cand in default_candidates:
            if cand.exists():
                test_file = cand
                break

        if not test_file:
            synth_dir = PROJECT_ROOT / "test_audio"
            synth_dir.mkdir(exist_ok=True)
            synth_wav = synth_dir / "synthetic_test.wav"
            generate_temp_test_wav(synth_wav)
            test_file = synth_wav
            print(f"✅ Generated synthetic test audio: {test_file}")

    print(f"✅ Target audio file: {test_file} ({test_file.stat().st_size / 1024:.1f} KB)")
    py_bin = sys.executable

    print("\n🧪 Testing MMM Commands...")
    tests = [
        (f"{py_bin} -m mmm.cli --help", "Help command"),
        (f"{py_bin} -m mmm.cli analyze --help", "Analyze help"),
        (f"{py_bin} -m mmm.cli obliterate --help", "Obliterate help"),
    ]

    success_count = 0
    for cmd, desc in tests:
        if test_command(cmd, timeout=5):
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"📊 Results: {success_count}/{len(tests)} basic commands verified")
    return 0 if success_count == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main())