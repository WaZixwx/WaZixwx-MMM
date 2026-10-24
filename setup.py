"""
Package configuration for WaZixwx-MMM Studio v7.0.0.
"""

from pathlib import Path
from setuptools import setup, find_packages

long_description = (
    Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""
)

setup(
    name="wazixwx-mmm-studio",
    version="7.0.0",
    description="WaZixwx-MMM Studio v7.0: Ultra-Transparent Golden-Ear Mastering, 7-Stem Separation & 10-Domain Forensic Workstation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WaZixwx",
    url="https://github.com/WaZixwx/mmm",
    license="PolyForm-Noncommercial-1.0.0",
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
            "flake8>=7.0.0",
            "mypy>=1.0.0",
            "pylint>=3.0.0",
        ],
        "gpu": [
            "onnxruntime-directml>=1.17.0",
            "audio-separator>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmm=mmm.cli:cli",
            "wazixwx-mmm=mmm.cli:cli",
            "mmm-studio=mmm.gui:launch_gui",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Mastering",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
