# Makefile for Melodic Metadata Massacrer

PYTHON ?= python3
PIP ?= pip3

.PHONY: install dev-install test test-unit test-cov lint format check-format clean build run examples test-audio demo ci docs help

# Default target
all: install

# Install package and dependencies
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# Install in development mode with test dependencies
dev-install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# Run tests with pytest (or fallback to unittest)
test:
	@which pytest > /dev/null 2>&1 && $(PYTHON) -m pytest tests/ -v --cov=mmm --cov-report=term || $(PYTHON) -m unittest discover -s tests -p "test_*.py"

# Run unittest suite directly
test-unit:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=mmm --cov-report=html

# Lint code
lint:
	flake8 mmm/ tests/ scripts/ examples/
	mypy mmm/

# Format code
format:
	black mmm/ tests/ scripts/ examples/
	isort mmm/ tests/ scripts/ examples/

# Check formatting
check-format:
	black --check mmm/ tests/ scripts/ examples/
	isort --check-only mmm/ tests/ scripts/ examples/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	$(PYTHON) setup.py sdist bdist_wheel

# Run the application
run:
	$(PYTHON) -m mmm.cli --help

# Example usage
examples:
	$(PYTHON) -m mmm.cli --help
	@echo "Example: mmm obliterate input.mp3 --paranoid --verify"

# Create test audio files for testing (self-contained wave generation)
test-audio:
	$(PYTHON) -c "
import wave, struct, math
from pathlib import Path

out_dir = Path('test_audio')
out_dir.mkdir(exist_ok=True)

def write_wav(filename, duration, freq, stereo=False):
    sr = 44100
    n = int(duration * sr)
    with wave.open(str(filename), 'wb') as w:
        w.setnchannels(2 if stereo else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            t = float(i) / sr
            s1 = int(16000 * math.sin(2.0 * math.pi * freq * t))
            if stereo:
                s2 = int(14000 * math.sin(2.0 * math.pi * (freq * 1.5) * t))
                frames.extend(struct.pack('<hh', s1, s2))
            else:
                frames.extend(struct.pack('<h', s1))
        w.writeframes(frames)

write_wav(out_dir / 'sine_440hz.wav', 2.0, 440.0, False)
write_wav(out_dir / 'stereo_test.wav', 2.0, 440.0, True)
print('Test audio files generated successfully in test_audio/')
"

# Demo with test files
demo: test-audio
	$(PYTHON) -m mmm.cli analyze test_audio/sine_440hz.wav
	$(PYTHON) -m mmm.cli obliterate test_audio/sine_440hz.wav --verify
	$(PYTHON) -m mmm.cli massacre test_audio/ --output-dir test_audio/cleaned/

# Release preparation
release-check: clean format lint test
	$(PYTHON) -m build

# Documentation
docs:
	$(PYTHON) -c "
import mmm
print('MMM Documentation')
print('==================')
print('Version:', mmm.__version__)
print('Description:', mmm.__description__)
print()
print('Commands:')
print('  mmm obliterate INPUT_FILE     - Remove all metadata and watermarks')
print('  mmm massacre DIRECTORY       - Batch process directory')
print('  mmm analyze INPUT_FILE        - Analyze without modifying')
print('  mmm config                   - Show configuration')
"

# Help
help:
	@echo "Available targets:"
	@echo "  install       - Install package and dependencies"
	@echo "  dev-install   - Install in development mode"
	@echo "  test          - Run tests (pytest or unittest fallback)"
	@echo "  test-unit     - Run unittest directly"
	@echo "  lint          - Lint code with flake8 and mypy"
	@echo "  format        - Format code with black and isort"
	@echo "  clean         - Remove build artifacts"
	@echo "  build         - Build package distribution"
	@echo "  test-audio    - Create self-contained synthetic test audio"
	@echo "  demo          - Run demo with test files"