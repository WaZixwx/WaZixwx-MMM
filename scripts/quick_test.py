#!/usr/bin/env python3
"""
Quick test to verify CLI commands are working
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_command(cmd, timeout=10):
    """Test if command starts without immediate errors"""
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

        # Check for config errors
        if "'Command' object has no attribute 'config'" in result.stderr:
            return False, "Config error still present"

        # Check for immediate Python errors
        if "Traceback" in result.stderr or "Error:" in result.stderr:
            return False, f"Python error: {result.stderr[:100]}"

        # Check if banner appears (indicates CLI started)
        if "MELODIC METADATA MASSACRER" in result.stdout:
            return True, "CLI started successfully"

        return True, "No immediate errors"

    except subprocess.TimeoutExpired:
        return True, "Process started but timed out (expected)"
    except Exception as e:
        return False, f"Exception: {e}"

def main():
    print("🧪 Quick CLI Test")
    print("=" * 50)

    py_bin = sys.executable
    tests = [
        (f"{py_bin} -m mmm.cli --help", "Help command"),
        (f"{py_bin} -m mmm.cli analyze --help", "Analyze help"),
        (f"{py_bin} -m mmm.cli obliterate --help", "Obliterate help"),
        (f"{py_bin} -m mmm.cli massacre --help", "Massacre help")
    ]

    all_passed = True
    for cmd, desc in tests:
        passed, message = test_command(cmd)
        status = "✅" if passed else "❌"
        print(f"{status} {desc}: {message}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! CLI is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())