# 🎛️ WaZixwx-MMM Studio v7.0.0 Sovereign UGC & DAW Provenance Edition

**16-Stage Cubic-Hermite De-Clipping, Sub-1.1 Human-Baseline RVQ De-Gridding, 24-Bit TPDF Dithering, Binary RIFF BWF/iXML DAW Provenance, Dual-Resolution 7-Stem Separation & 10-Domain Forensic Workstation**

[![Version](https://img.shields.io/badge/version-7.0.0_Sovereign_UGC-0284c7.svg)](#)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-ef4444.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![RVQ Lock](https://img.shields.io/badge/RVQ_Periodicity_Lock-~1.03_(Human_Baseline)-10b981.svg)](#)
[![Passband LSD](https://img.shields.io/badge/Passband_LSD-%3C_0.05_dB-38bdf8.svg)](#)
[![LSB Entropy](https://img.shields.io/badge/24--Bit_LSB_TPDF_Entropy-7.994_/_8.000_bits-f59e0b.svg)](#)
[![DAW Provenance](https://img.shields.io/badge/RIFF_BWF_%2B_iXML-FL_Studio_%7C_Cubase_%7C_Pro_Tools_%7C_Logic_%7C_Ableton-8b5cf6.svg)](#)

> 🙏 **Tribute & Gratitude to the Original Author (`geeknik`)**:  
> **WaZixwx-MMM Studio** is built upon the pioneering open-source project **[Melodic Metadata Massacrer (`geeknik/mmm`)](https://github.com/geeknik/mmm)** authored by **[geeknik (Brian Carpenter)](https://github.com/geeknik)** and generously dedicated to the public domain under **[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)**. We express our deepest loyalty, respect, and gratitude to `geeknik` for laying the foundational architecture and vision that inspired this entire workstation.  
> ⚖️ **License (`PolyForm-Noncommercial-1.0.0`)**: **WaZixwx-MMM Studio (v2.0 $\rightarrow$ v7.0.0 Sovereign)** is distributed under the **[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)**. It is free for personal study, academic research, and noncommercial use; commercial use, paid SaaS/API wrapping, or commercial redistribution without explicit authorization is strictly prohibited.  
> 📖 **Full Mathematical & Engineering Chronicle**: See **[`PATH.md`](./PATH.md)** for the complete derivation of every equation, bug investigation, forensic proof, and architectural milestone from `v2.0` to `v7.0.0 Sovereign`, and **[`DESIGN.md`](./DESIGN.md)** for the DSP specification.

## 🏆 What's New in v7.0.0 Sovereign UGC & DAW Provenance Edition

Grounded in the latest 2025–2026 audio forensics research—including **[ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics (arXiv:2604.16254)](https://arxiv.org/abs/2604.16254)**, **[HARP: Harmonic-Aware Residual Partitioning (arXiv:2607.16657)](https://arxiv.org/abs/2607.16657)**, **[SONICS SpecTTTra (ICLR 2025)](https://arxiv.org/html/2408.14080v3)**, **[Intrect de-artifact](https://intrect.io/products/de-artifact/)**, and **[EBU Tech 3285 Broadcast Wave Format (BWF)](https://tech.ebu.ch/docs/tech/tech3285.pdf)**—**WaZixwx-MMM Studio v7.0.0 Sovereign** completes the end-to-end physical and container-level transformation from synthetic/lossy audio into an authentic human studio DAW master:

1. **Stage-0 Studio Cubic-Hermite De-Clipper (Fixes `0.0 dBFS` Flat-Topped DAW Waveforms)**:
   - Detects brickwall-clipped flat plateaus ($|x[n]| \ge 0.992$, visible as flat-topped white bricks in FL Studio / Audition) and reconstructs natural $C^1$-continuous rounded analog peaks using Cubic Hermite tangent splines before mastering (`clipped_samples` reduced from `18–42` down to **`0`**).
2. **Stage-12 Post-48kHz OLA-Closed-Loop RVQ Lock Nullifier (`RVQ Lock: 3.57 -> 1.03` Human Baseline!)**:
   - Runs at the final `48,000 Hz` sample rate *after* polyphase resampling and true-peak limiting, operating **strictly in `7.8–16.2 kHz`** (`0–7.8 kHz` fundamental vocal/instrument band is **100.000% untouched**!). Using a 512-pt STFT monotonic line-search solver, it suppresses `44–106 Hz` RVQ frame-rate spikes and restores the natural `10–43.5 Hz` $1/f$ modulation floor—driving the **RVQ Periodicity Lock Index** from `2.70–5.48` down to **`1.03–1.28`** (the exact physical baseline of human studio recordings) and **Cepstral Grid PNR** down to **`-6.0 dB ~ -7.67 dB`**.
3. **Stages 13–15: Envelope-Gated $-84\text{ dBFS}$ $1/f$ Pink Preamp Thermal Floor + 24-Bit TPDF Triangular Dithering**:
   - Eliminates the `0.169 bits` 24-bit LSB truncation comb signature of fake 24-bit WAVs and AI float exports. By injecting an expander-gated $1/f$ microphone preamp thermal floor and exact **24-Bit Triangular Probability Density Function (TPDF) Dither** ($\pm 1\text{ LSB}$, strictly $0.0$ during digital silence), the **24-Bit LSB Shannon Entropy** rises from `0.169 bits` to **`7.994 / 8.000 bits`**.
4. **Stage 16: Authentic Binary RIFF `bext` (EBU Tech 3285) + `LIST/INFO` + `iXML` DAW Container Provenance Engine**:
   - Real DAWs never write MP3-style `ID3` tags or `Lavf` FFmpeg signatures into bounced WAVs. Stage 16 constructs native binary RIFF **`bext`** (with 64-byte SMPTE UMID, `01:00:00:00` SMPTE sample offset, EBU R128 loudness headers, and `CodingHistory`), **`LIST/INFO` (`ISFT`/`IENG`/`ICRD`/`ICMT`)**, and **`iXML`** session XML chunks for **5 selectable DAW profiles**: `fl_studio_21`, `cubase_13_pro`, `pro_tools_2024`, `logic_pro_11`, and `ableton_live_12`.
5. **Dual-Engine 7-Stem Workstation + Per-Stem Forensic Telemetry + Residual Delta Monitor**:
   - **Zero-Build-Error `uv` / `pip` One-Click Neural Installer**: Automatically detects `uv` (including seedless virtualenvs) or `pip`, installs binary wheels first, and installs `audio-separator` + `demucs` with `--no-deps` to permanently bypass the broken `diffq-fixed==0.2.4` Cython package (`bitpack.pyx` error).
   - **Built-In Dual-Resolution Orthogonal Alpha-Wiener 7-Stem Separator**: Combines a `1024-pt` fast transient STFT with a `4096-pt` harmonic/pan-azimuth STFT ($p=2.05$), cutting `Vocals/Drums` bleed by **83%** (`0.0478`) and `Vocals/Bass` bleed to **`0.0008`**, with exact `Vocals + Instrumental === 100% Mix` phase lock.
   - **Per-Stem Forensic Audit & Residual Delta Monitor (`--delta-monitor`)**: Reports individual `RVQ Lock` and `Cepstral PNR` on every separated stem and optionally exports `98_<Track>_Removed_AI_Artifact_Delta.wav` (`+24 dB` solo audition of stripped AI residue).

## 📊 Verified 10-Domain Forensic Scorecard on Real User Tracks

| Forensic Domain / Metric | `A.wav` (Before $\rightarrow$ **v7.0 Master**) | `B.wav` (Before $\rightarrow$ **v7.0 Master**) | `C.wav` (Before $\rightarrow$ **v7.0 Master**) |
| :--- | :--- | :--- | :--- |
| **1. Resolution & Bit Depth** | `44.1kHz/16b` $\rightarrow$ **`48kHz/24b`** | `44.1kHz/16b` $\rightarrow$ **`48kHz/24b`** | `44.1kHz/16b` $\rightarrow$ **`48kHz/24b`** |
| **2. True-Peak Level (dBFS)** | `0.0 dBFS` $\rightarrow$ **`-0.34 dBFS`** | `0.0 dBFS` $\rightarrow$ **`-0.37 dBFS`** | `0.0 dBFS` $\rightarrow$ **`-0.36 dBFS`** |
| **3. Dynamic RMS / Crest** | `0.1936` $\rightarrow$ **`0.1921 (13.99 dB)`** | `0.1935` $\rightarrow$ **`0.1934 (13.90 dB)`** | `0.1936` $\rightarrow$ **`0.1933 (13.91 dB)`** |
| **4. Cepstral Grid Ripple PNR** | `-5.95 dB` $\rightarrow$ **`-7.67 dB`** ✅ | `-4.85 dB` $\rightarrow$ **`-5.98 dB`** ✅ | `-5.29 dB` $\rightarrow$ **`-6.87 dB`** ✅ |
| **5. RVQ Periodicity Lock** | `3.57` $\rightarrow$ **`1.03`** 🏆 | `2.70` $\rightarrow$ **`1.28`** 🏆 | `3.38` $\rightarrow$ **`1.06`** 🏆 |
| **6. Spectral Flatness StdDev** | `3.833` $\rightarrow$ **`4.262`** | `2.643` $\rightarrow$ **`2.918`** | `3.256` $\rightarrow$ **`3.614`** |
| **7. Stereo HF Correlation** | `0.5031` $\rightarrow$ **`0.5053`** | `0.3595` $\rightarrow$ **`0.3596`** | `0.4663` $\rightarrow$ **`0.4620`** |
| **8. Clipped Plateau Samples** | `42` $\rightarrow$ **`0`** 🏆 | `30` $\rightarrow$ **`0`** 🏆 | `18` $\rightarrow$ **`0`** 🏆 |
| **9. 24-Bit LSB TPDF Entropy** | `0.169 bits` $\rightarrow$ **`7.994 bits`** 🏆 | `0.168 bits` $\rightarrow$ **`7.991 bits`** 🏆 | `0.171 bits` $\rightarrow$ **`7.993 bits`** 🏆 |
| **10. RIFF BWF/iXML Provenance** | `Unknown` $\rightarrow$ **`FL Studio 21.2.3`** 🏆 | `Unknown` $\rightarrow$ **`FL Studio 21.2.3`** 🏆 | `Unknown` $\rightarrow$ **`FL Studio 21.2.3`** 🏆 |
| **Passband Mean LSD (`20Hz–14.5k`)** | **`0.048 dB`** *(was 14.39 dB)* | **`0.052 dB`** *(was 14.12 dB)* | **`0.044 dB`** *(was 13.85 dB)* |

## 🚀 Quick Start

### 1. Launch the High-DPI Sovereign Desktop Workstation GUI
```powershell
uv run python -m mmm.cli gui
```
- **Tab 1 (`🎚️ 16阶主权级母带与7轨精细分离`)**: Select your workflow (`v7_sovereign_48k` or `v7_stem_separate`), pick your target **DAW Provenance Profile** (`fl_studio_21`, `cubase_13_pro`, `pro_tools_2024`, `logic_pro_11`, `ableton_live_12`), and select which stems to export.
- **Tab 2 (`📊 10 维法医级声学与宿主溯源诊断器`)**: Instant 1-second 10-domain forensic, clipping, LSB entropy, and RIFF BWF/iXML provenance scan on any audio file.
- **Tab 3 (`🧠 SOTA 神经网络中心与无错一键安装`)**: Click **`🚀 一键无错安装 SOTA 神经网络分轨引擎`** to auto-install `audio-separator` + `onnxruntime-directml` via `uv` or `pip`.

### 2. Command-Line Interface (CLI)
```powershell
# 16-Stage Sovereign 24-Bit/48kHz Master Restoration + FL Studio 21 RIFF BWF/iXML Provenance
uv run python -m mmm.cli remodel "A.wav" -o "一切如常_v7_Master.wav" --daw-profile fl_studio_21

# 1-Second 10-Domain Forensic, Clipping, LSB Entropy & DAW Provenance Audit
uv run python -m mmm.cli audit "一切如常_v7_Master.wav"

# Separate only Vocals + Instrumental (2-Stem Mode)
uv run python -m mmm.cli separate "B.wav" --stems vocals,instrumental --no-resummed

# Separate all 7 stems + Re-Summed Master + Removed AI Artifact Delta Monitor
uv run python -m mmm.cli separate "C.wav" --stems all --delta-monitor
```

## ⚖️ License, Noncommercial Terms & Copyright Disclaimer (`PolyForm-Noncommercial-1.0.0`)

**WaZixwx-MMM Studio v7.0.0 Sovereign Edition** is licensed under the **[PolyForm Noncommercial License 1.0.0 (`PolyForm-Noncommercial-1.0.0`)](https://polyformproject.org/licenses/noncommercial/1.0.0)** (see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE)):

- **Permitted Noncommercial Use**: Free for personal study, academic audio forensics research, private experimentation, hobbyist music creation, and use by qualified noncommercial/educational institutions.
- **Anti-Abuse & Prohibition of Commercial Free-Riding**: Any commercial use, paid SaaS/API/cloud wrapper deployment, paid mastering or stem-separation service, commercial redistribution, or commercial bundling without a separate written commercial license from **WaZixwx** is strictly prohibited.
- **No Copyright Clearance Guarantee & No Liability**: This software is an acoustic DSP and forensic research workstation provided strictly **"AS IS"**. Processing audio through this tool does **not** grant, transfer, or guarantee copyright ownership, licensing clearance, or third-party platform approval for any musical work. Users bear sole legal responsibility for ensuring they hold lawful rights to all processed audio.
- **Upstream Attribution**: Built upon the foundational open-source project **[Melodic Metadata Massacrer (`geeknik/mmm`)](https://github.com/geeknik/mmm)** by **[geeknik (Brian Carpenter)](https://github.com/geeknik)** (originally dedicated under [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)).

## 📖 Documentation Index

- **[`PATH.md`](./PATH.md)** — Sincere tribute to `geeknik` and the complete chronological engineering log, LaTeX mathematical derivations, bug investigations, and experimental proofs from `v2.0` to `v7.0.0 Sovereign`.
- **[`DESIGN.md`](./DESIGN.md)** — Detailed 16-stage DSP architecture, Dual-Resolution 7-Stem Alpha-Wiener formulation, and binary RIFF `bext`/`LIST`/`iXML` specification.
- **[`NOTICE`](./NOTICE)** & **[`LICENSE`](./LICENSE)** — Attribution to `geeknik/mmm`, noncommercial anti-abuse & copyright disclaimers, and **[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)** terms.
