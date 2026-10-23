from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""

setup(
    name="melodic-metadata-massacrer",
    version="2.0.1",
    description="The audio anonymizer that makes AI detectors cry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Security Research Team",
    url="https://github.com/geeknik/mmm",
    packages=find_packages(include=["mmm", "mmm.*"]),
    install_requires=[
        "click>=8.1.0",
        "mutagen>=1.47.0",
        "librosa>=0.10.1",
        "pydub>=0.25.1",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.1",
        "colorama>=0.4.6",
        "rich>=13.0.0",
        "PyYAML>=6.0",
        "flask>=3.0.0,<4.0.0",
        "soxr>=0.3.0",
        "numba>=0.59.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "gpu": [
            "cupy>=12.0.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmm=mmm.cli:cli",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
