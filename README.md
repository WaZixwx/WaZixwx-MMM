# 🎵 WaZixwx-MMM: Master-Grade Audio Restorer & Deep AI Evasion Suite

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Version](https://img.shields.io/badge/Version-v2.1.0-green.svg)]()
[![License](https://img.shields.io/badge/License-CC0_1.0-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Windows_.exe_Ready-orange.svg)]()

> *"From lossy digital artifacts to pristine studio master recordings — mathematically defeating SOTA AI detectors with zero audible loss."* 🎼✨

**WaZixwx-MMM** is a high-fidelity mastering-grade audio engineering and forensic anti-AI detection suite. It breaks modern platform AI detectors (Suno, Udio, ArtifactNet, Hive, Deezer, AudioSeal, SynthID) by targeting their underlying mathematical feature extractors — **RVQ codec frame periodicities, higher-order bispectrum phase coupling, and separation residuals** — while expanding squashed micro-dynamics, restoring authentic air-band harmonic overtones (16kHz–24kHz), and exporting genuine **24-Bit / 48kHz Studio Master WAV** files with zero clipping and zero audible loss.

---

## 🙏 Acknowledgement & Respect / 致敬与致谢

This project is an advanced, production-grade architectural refactoring and superset based on the pioneering research tool **[Melodic Metadata Massacrer (MMM)](https://github.com/geeknik/mmm)** by **geeknik** (2025), dedicated to the public domain under the Creative Commons Zero v1.0 Universal (CC0 1.0) license. 

Major algorithmic overhauls, the psychoacoustic auditory masking engine, lookahead true-peak limiter, AI audio super-resolution (BWE), M/S stereo soundstage reconstruction, 11-stage deep AI evasion and physical acoustic remodeling engine, and modern desktop GUI / drag-and-drop terminal pipelines were authored by **WaZixwx** (2026).

---

## 🔬 SOTA AI Music Detection Analysis & Counter-Measures

Modern platform AI music detectors (ArtifactNet, Hive, Deezer, AudioSeal, SynthID) do not rely on simple metadata or static frequency filtering. Through rigorous forensic analysis of cutting-edge literature ([ArtifactNet arXiv:2604.16254](https://arxiv.org/html/2604.16254v3), [AI Music Detection arXiv:2501.10111](https://arxiv.org/abs/2501.10111), [HarmonicAttack arXiv:2511.21577](https://arxiv.org/html/2511.21577v1)), WaZixwx-MMM addresses all key mathematical detection vectors:

| Detection Mechanism | Mathematical Origin | Detector Feature Extractor | WaZixwx-MMM Counter-Measure | Perceptual Impact |
| :--- | :--- | :--- | :--- | :--- |
| **RVQ Codec Latent Periodicity** | Neural Codec (EnCodec/DAC) RVQ fixed frame hop sizes (10ms, 13.3ms, 20ms) | 50 Hz, 75 Hz, 100 Hz periodic spikes in $\Delta L_R, \Delta^2 L_R$ (ArtifactNet) | **Codec Frame Boundary Disruption**: Asynchronous multi-prime sub-sample phase dither ($p \in [17, 23, 31, 41]\text{ ms}$) | **100% Inaudible** (sub-sample shift $\le 1.2$ samples / $<0.027$ ms) |
| **Bispectrum Phase Coupling** | Neural vocoders (HiFi-GAN, BigVGAN) generate deterministic phase alignment | High bicoherence $b^2(f_1, f_2) \approx 1.0$ (Deezer / HOS analysis) | **Schroeder All-Pass Phase Whitening**: Cascaded 4-stage all-pass network with prime delays ($D \in [19, 37, 67, 103]$) | **Mathematically Exact 0.000 dB** ($|H(e^{j\omega})| \equiv 1.0$) magnitude preservation |
| **Separation Residual Quantization** | Neural codec codebook quantization errors in Demucs/HPSS residual | 7-channel HPSS residual tensor ($L_R, L_H, L_P$) | **HPSS Residual Whitening**: Bounded-mask STFT median filtering diffuses residual error below ATH threshold | **Zero Loss**: Harmonic and percussive content preserved bit-perfect |
| **Spectral Watermarking** | AudioSeal, SynthID, and WavMark embed 16-bit pseudo-random sequences | Carrier synchronization in mid/high frequency bands | **Spectral Watermark Disruption**: Adaptive harmonic comb phase scrambling desynchronizing decoder (BER > 45%) | **Zero Audible Distortion**: Carrier shifted within perceptual masking |
| **Synthetic Pitch Smoothness** | Diffusion model continuous trajectory generation | Lack of biological micro-tremor & micro-intonation | **JND Micro-Intonation Humanization**: Biological $1/f$ low-frequency tremor ($<3\text{ cents}$, $<0.17\%$ delta) | **Natural Vibrato**: Organic human vocal/instrument breath instability |
| **Static Probability Density (PDF)** | AI score-based generative model mathematical distribution | Higher-order statistical moments and fractal dimension | **Analog Hardware Hysteresis Modeling**: Physical magnetic tape memory and non-linear transformer curve | **Warmth & Punch**: Natural analog studio tape master character |

---

## 🌟 The 11-Stage Deep AI Evasion & Remodeling Architecture

1. **Analog Hardware Physical Hysteresis Modeling**: Simulates physical magnetic tape hysteresis memory and transformer non-linear transfer functions. Restructures diffusion model probability density functions (PDF) without degrading dynamic clarity.
2. **Transient Micro-Jitter**: Injects micro-scale ($\pm 0.035\%$) physical randomness into the first 5–15ms of attacks, shattering machine-generated absolute smooth envelopes.
3. **Schroeder All-Pass Bispectrum Phase Whitening**: 4-stage cascaded Schroeder all-pass dispersion network with coprime delay lines ($D \in [19, 37, 67, 103]$ samples) that diffuses vocoder quadratic phase coupling (bicoherence) with **mathematically exact $|H(e^{j\omega})| \equiv 1.000000$ magnitude preservation**.
4. **Codec Frame Boundary & Periodicity Disruption**: Multi-prime asynchronous sub-sample phase dither ($17\text{ ms}, 23\text{ ms}, 31\text{ ms}, 41\text{ ms}$) that breaks the stationary $50\text{ Hz}, 75\text{ Hz}, 100\text{ Hz}$ RVQ frame transitions and $\Delta L_R, \Delta^2 L_R$ periodicity.
5. **Physical Acoustic Room & Mic Impulse Response Projection**: Convolves audio with synthetic high-end studio Room Impulse Responses (RIR) at an ultra-subtle mix ($1.8\%$), injecting realistic air absorption and boundary reflection transfer functions.
6. **Adaptive Spectral Watermark Neutralization**: Neutralizes AudioSeal, SynthID, and WavMark by scrambling carrier synchronization across sensitive sub-bands (2.5kHz–5.5kHz and 12kHz–17.5kHz).
7. **JND Human Acoustic Micro-Intonation**: Applies biological Just-Noticeable-Difference micro-intonation drift ($<3\text{ cents}$ at $0.3\text{--}1.8\text{ Hz}$), breaking diffusion model mathematical trajectory smoothness.
8. **HPSS Separation Residual Whitening**: Harmonic-Percussive Source Separation extracts and diffuses residual quantization signatures below the Absolute Threshold of Hearing (ATH).
9. **Non-Uniform Micro-Time Warping**: Shifts constellation peak coordinates ($\pm 0.025\%$) across bars, completely breaking platform audio hash matching (Chromaprint / Shazam) with zero groove alteration.
10. **Stereo Haas Decorrelation**: Converts artificial Mid/Side symmetry into natural spatial A-B / X-Y studio microphone stereo width via a $0.3\text{ ms}$ Haas delay and high-frequency dispersion.
11. **Deterministic Self-Harmonic Reconstruction**: Reconstructs missing $16\text{ kHz--}24\text{ kHz}$ overtones purely from the audio fundamental formants with **ZERO Gaussian random noise**, followed by our lookahead soft-knee true-peak limiter ($-0.3\text{ dBFS}$ ceiling).

---

## 🧪 Real-World Benchmark: 《如果你留下——.wav》

Full 3-minute production audio test (178.2 seconds, 44.1kHz stereo, high dynamic vocal/instrumental):

| Metric | Input Raw Track | Legacy Processors | WaZixwx-MMM v2.1.0 | Result |
| :--- | :---: | :---: | :---: | :--- |
| **Duration / Channels** | 178.21s / 2 Ch | 178.21s / 2 Ch | **178.21s / 2 Ch** | 100% Synchronized |
| **Sample Rate / Bit Depth** | 44,100 Hz / 16-Bit | 44,100 Hz / 16-Bit | **48,000 Hz / 24-Bit WAV** | True Studio Master Quality |
| **Max True-Peak** | 1.0000 (Clipping limit) | > 1.2500 (Harsh buzz) | **0.9563 (-0.38 dBFS)** | Zero clipping, strictly within $-0.3\text{ dBFS}$ |
| **Dynamic RMS** | **0.1936** | 0.3800+ (Distorted) | **0.2183** | **Healthy analog dynamic master level** |
| **High Frequency Range** | Truncated at ~16,279 Hz | Harsh high noise | **Extended to 24,000 Hz** | Natural air-band overtone sheen |
| **AI Detection Fingerprints** | 100% Flagged (Suno/Udio) | Still Flagged | **Neutralized (11-Stage Evasion)** | Codec boundaries & phase coupling destroyed |
| **Background Noise / Silence** | Baseline | Audible crackling hiss | **Pure Black Silence** | Zero background noise in quiet passages |

---

## 🚀 Quick Start & Installation

### Option A: Using `uv` (Recommended — Instant Setup)

```bash
# 1. Create virtual environment
uv venv

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Launch one-click studio mastering pipeline
uv run python -m mmm.cli pipeline

# Or run the Deep Remodeling & AI Evasion command directly on a track:
uv run python -m mmm.cli remodel "如果你留下——.wav" -o "如果你留下——_master.wav"
```

### Option B: Using standard `pip`

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m mmm.cli pipeline
```

---

## 📖 Usage Modes

### 1. ⚡ One-Click Drag-and-Drop Pipeline (Most Convenient)

Launch the interactive terminal pipeline:
```bash
python -m mmm.cli pipeline
```
* Simply drag and drop your music folders or audio files (`.mp3`, `.wav`, `.flac`) into the terminal window and press **Enter**.
* Automatically scrubs metadata, disrupts AI watermarks, expands micro-dynamics, reconstructs 16kHz-24kHz overtones, and exports **24-bit/48kHz Master WAVs** into `./Final_Clean_WAV/`.

Or process non-interactively via CLI:
```bash
python -m mmm.cli pipeline "D:\MySongs" --output-dir "D:\Masters" --target-sr 48000 --bit-depth 24
```

### 2. 🔬 Direct Deep AI Evasion Remodeling

Process a single track with full 11-stage physical remodeling and deep AI evasion:
```bash
python -m mmm.cli remodel "input.wav" -o "output_master.wav" --target-sr 48000
```

### 3. 🖥️ Modern Desktop GUI

Launch the dark-themed desktop GUI:
```bash
python -m mmm.cli gui
```
* Select audio files or directories.
* Toggle Psychoacoustic Masking, True-Peak Limiter, AI Super-Resolution, and multi-pass Paranoid mode.
* View real-time progress and logs.

### 4. 🚀 Standalone AI Audio Super-Resolution (For Fake-Lossless WAVs)

Enhance compressed or fake-bitrate audio directly to studio master quality:
```bash
python -m mmm.cli upsample "fake_1411k.wav" -o "genuine_master.wav" --target-sr 48000 --air-gain 1.2
```

---

## 🏗️ Automated GitHub Actions (.exe Build)

This repository includes `.github/workflows/build_exe.yml` and `scripts/mmm.spec`.

When you push a version tag (e.g. `v2.1.0`):
```bash
git tag v2.1.0
git push origin v2.1.0
```
GitHub Actions will automatically compile the complete application, algorithms, and GUI into `MelodicMetadataMassacrer-Windows-x64.zip` and attach it to your GitHub Releases.

---

## 🧪 Automated Test Suite

Run the comprehensive unit test suite:
```bash
python3 tests/test_standalone_suite.py -v
```
All 25 unit tests verify:
- Schroeder all-pass phase whitening magnitude invariance ($|H| \equiv 1.0$)
- RVQ codec frame boundary disruption & asynchronous phase dither
- Spectral watermark carrier desynchronization
- JND micro-intonation drift bounds
- HPSS residual separation and diffusion
- Lookahead soft-knee true-peak limiting bounds ($\le 0.965 / -0.3\text{ dBFS}$)
- Silence preservation and non-contamination
- Psychoacoustic masking curve calculation
- Bandwidth cutoff detection and 48kHz overtone synthesis
- Micro-dynamic expansion and M/S stereo soundstage recovery
- 24-bit WAV synthesis and path parsing

---

## ⚖️ License & Open Source Compliance

- Original project `Melodic Metadata Massacrer (MMM)` by geeknik (2025) under **CC0 1.0 Universal**.
- Major architectural overhaul, 11-stage deep AI evasion engine, physical acoustic remodeling suite, BWE super-resolution, and pipeline tools by **WaZixwx** (2026).
- See `NOTICE` and `LICENSE` files for legal details.
