"""
Comprehensive standalone unit test suite for MMM.
Tests CLI, configuration, DSP math, channel normalization, psychoacoustics, and synthetic audio IO.
"""

import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure optional heavy dependencies don't block unit testing
for mod_name in ["librosa", "soundfile", "flask"]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

if "mutagen" not in sys.modules:
    try:
        import mutagen
    except ImportError:
        mutagen_mock = MagicMock()
        mutagen_mock.__path__ = []
        sys.modules["mutagen"] = mutagen_mock
        for sub in ["mp3", "id3", "flac", "wave"]:
            m = MagicMock()
            sys.modules[f"mutagen.{sub}"] = m
            setattr(mutagen_mock, sub, m)

import numpy as np
from click.testing import CliRunner

from mmm.preserving_sanitizer import AudioSuperResolution
from scipy import signal
from mmm.preserving_sanitizer import MasterGradeAudioRestorer
from mmm.preserving_sanitizer import parse_drag_drop_paths, scan_audio_files
from mmm import __version__
from mmm.cli import cli
from mmm.config.config_manager import ConfigManager
from mmm.config.defaults import PRESETS, DEFAULT_CONFIG
from mmm.ui.banners import BannerManager
from mmm.ui.console import ConsoleManager
from mmm.preserving_sanitizer import _repair_non_finite_audio, _ensure_channel_layout
from mmm.preserving_sanitizer import (
    true_peak_limiter,
    calculate_masking_threshold,
    apply_psychoacoustic_perturbation,
)


class TestMMMCLI(unittest.TestCase):
    """Test CLI commands and options."""

    def setUp(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Melodic Metadata Massacrer", result.output)
        self.assertIn("obliterate", result.output)
        self.assertIn("analyze", result.output)
        self.assertIn("massacre", result.output)

    def test_version_option(self):
        result = self.runner.invoke(cli, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(__version__, result.output)

    def test_version_command(self):
        result = self.runner.invoke(cli, ["version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(__version__, result.output)

    def test_config_display(self):
        result = self.runner.invoke(cli, ["config"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Configuration", result.output)


class TestConfigManager(unittest.TestCase):
    """Test configuration loading, validation, and preset application."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.temp_dir.name) / "config.yaml"
        self.config_manager = ConfigManager(config_file=self.config_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_presets_exist(self):
        for preset_name in ["stealth", "fast", "quality", "stealth-plus"]:
            self.assertIn(preset_name, PRESETS)
            self.assertIsInstance(PRESETS[preset_name], dict)

    def test_default_config_structure(self):
        config = self.config_manager.get_config()
        self.assertIn("metadata_cleaning", config)
        self.assertIn("spectral_cleaning", config)
        self.assertIn("output_format", config)

    def test_get_and_set_config(self):
        self.config_manager.set("test_key", "custom_val")
        self.assertEqual(self.config_manager.get("test_key"), "custom_val")
        self.config_manager.save_config()
        self.assertTrue(self.config_file.exists())

    def test_preset_management(self):
        preset_name = "test_custom_preset"
        test_data = {"test_mode": True, "value": 123}
        self.config_manager.create_preset(preset_name, test_data)
        loaded = self.config_manager.load_preset(preset_name)
        self.assertEqual(loaded.get("value"), 123)
        self.config_manager.delete_preset(preset_name)


class TestPsychoacousticSanitizer(unittest.TestCase):
    """Test auditory masking, silence preservation, and true-peak limiting."""

    def test_true_peak_limiter_bounds(self):
        hot_signal = np.array([-3.5, -1.2, 0.0, 0.5, 1.8, 2.5], dtype=np.float32)
        limited = true_peak_limiter(hot_signal, target_peak=0.965)
        self.assertLessEqual(np.max(np.abs(limited)), 0.965)
        self.assertTrue(np.all(np.isfinite(limited)))

    def test_silence_preservation(self):
        """Ensure silence is NEVER contaminated with noise or dither."""
        sr = 44100
        silence = np.zeros(44100, dtype=np.float32)
        sanitized = apply_psychoacoustic_perturbation(silence, sr=sr, silence_threshold=1e-4)
        self.assertEqual(np.max(np.abs(sanitized)), 0.0)

    def test_masking_threshold_curve(self):
        sr = 44100
        mags = np.ones(1025, dtype=np.float32) * 0.1
        curve = calculate_masking_threshold(mags, sr)
        self.assertEqual(len(curve), 1025)
        self.assertTrue(np.all(curve > 0.0))


class TestAudioDSPAndNormalization(unittest.TestCase):
    """Test core DSP operations, NaN protection, and channel layouts."""

    def test_repair_non_finite_audio(self):
        data = np.array([0.0, 0.5, np.nan, -0.8, np.inf, -np.inf, 1.5, -2.0], dtype=np.float32)
        repaired = _repair_non_finite_audio(data, label="test")
        self.assertTrue(np.all(np.isfinite(repaired)))
        self.assertEqual(repaired[2], 0.0)
        self.assertEqual(repaired[4], 1.0)
        self.assertEqual(repaired[5], -1.0)
        self.assertEqual(repaired[6], 1.0)
        self.assertEqual(repaired[7], -1.0)

    def test_ensure_channel_layout_mono(self):
        mono_1d = np.zeros(100, dtype=np.float32)
        res = _ensure_channel_layout(mono_1d)
        self.assertEqual(res.shape, (100, 1))

    def test_ensure_channel_layout_stereo(self):
        stereo_channels_first = np.zeros((2, 500), dtype=np.float32)
        res = _ensure_channel_layout(stereo_channels_first)
        self.assertEqual(res.shape, (500, 2))
        self.assertTrue(res.flags.c_contiguous)

    def test_synthetic_wav_io(self):
        """Test synthesizing and reading a 16-bit stereo WAV file."""
        sample_rate = 44100
        duration = 0.5
        n_samples = int(sample_rate * duration)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test_synth.wav"

            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                frames = bytearray()
                for i in range(n_samples):
                    t = float(i) / sample_rate
                    left = int(10000 * math.sin(2.0 * math.pi * 440.0 * t))
                    right = int(10000 * math.cos(2.0 * math.pi * 440.0 * t))
                    frames.extend(struct.pack("<hh", left, right))
                wf.writeframes(frames)

            self.assertTrue(wav_path.exists())

            with wave.open(str(wav_path), "rb") as rf:
                self.assertEqual(rf.getnchannels(), 2)
                self.assertEqual(rf.getsampwidth(), 2)
                self.assertEqual(rf.getframerate(), sample_rate)
                self.assertEqual(rf.getnframes(), n_samples)
                raw_data = rf.readframes(n_samples)
                self.assertEqual(len(raw_data), n_samples * 4)


class TestUIComponents(unittest.TestCase):
    """Test console logging and banners."""

    def test_banner_generation(self):
        banner = BannerManager()
        self.assertIsNotNone(banner.main_banner)
        self.assertIn("WaZixwx-MMM STUDIO", banner.main_banner)

    def test_console_manager(self):
        console = ConsoleManager()
        console.info("Testing info message")
        console.success("Testing success message")
        console.warning("Testing warning message")
        console.error("Testing error message")



class TestAudioSuperResolution(unittest.TestCase):
    """Test AI Audio Super-Resolution and Bandwidth Extension."""

    def setUp(self):
        self.asr = AudioSuperResolution(target_sample_rate=48000, air_band_gain=1.0)

    def test_detect_bandwidth_cutoff(self):
        sr = 44100
        np.random.seed(42)
        white = np.random.normal(0, 0.5, sr)
        b, a = signal.butter(8, 12000.0 / (sr / 2), btype='low')
        filtered = signal.filtfilt(b, a, white)
        cutoff = self.asr.detect_bandwidth_cutoff(filtered, sr)
        self.assertGreaterEqual(cutoff, 10000)
        self.assertLessEqual(cutoff, 16000)

    def test_enhance_audio(self):
        sr = 22050
        t = np.linspace(0, 0.5, sr // 2)
        sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        enhanced, new_sr = self.asr.enhance_audio(sig, sr)
        self.assertEqual(new_sr, 48000)
        self.assertLessEqual(np.max(np.abs(enhanced)), 0.965)
        self.assertTrue(np.all(np.isfinite(enhanced)))

    def test_super_resolve_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_path = Path(tmp_dir) / "input_low.wav"
            out_path = Path(tmp_dir) / "output_hd.wav"
            sr = 22050
            n = sr // 2
            with wave.open(str(in_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                frames = bytearray()
                for i in range(n):
                    val = int(12000 * math.sin(2 * math.pi * 440 * (i / sr)))
                    frames.extend(struct.pack("<h", val))
                wf.writeframes(frames)

            res = self.asr.super_resolve_file(in_path, out_path)
            self.assertTrue(res["success"])
            self.assertEqual(res["target_sr"], 48000)
            self.assertTrue(out_path.exists())



class TestMasterGradePipeline(unittest.TestCase):
    """Test Master-Grade Audio Restoration Engine and Pipeline Utilities."""

    def setUp(self):
        self.restorer = MasterGradeAudioRestorer(target_sample_rate=48000, bit_depth=24)

    def test_dequantize_and_dynamics(self):
        sr = 44100
        t = np.linspace(0, 0.5, sr // 2)
        audio = np.column_stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 440 * t)])
        expanded = self.restorer.dequantize_and_expand_dynamics(audio, sr)
        self.assertEqual(expanded.shape, audio.shape)
        self.assertLessEqual(np.max(np.abs(expanded)), 0.965)

    def test_stereo_soundstage(self):
        sr = 44100
        t = np.linspace(0, 0.5, sr // 2)
        audio = np.column_stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 880 * t)])
        enhanced = self.restorer.restore_stereo_soundstage(audio, sr)
        self.assertEqual(enhanced.shape, audio.shape)
        self.assertLessEqual(np.max(np.abs(enhanced)), 0.965)

    def test_restore_file_to_master_wav(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_wav = Path(tmp_dir) / "input_track.wav"
            out_master = Path(tmp_dir) / "Final_Clean_WAV" / "output_master.wav"

            sr = 44100
            n = sr // 2
            with wave.open(str(in_wav), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                frames = bytearray()
                for i in range(n):
                    l_val = int(14000 * math.sin(2 * math.pi * 440 * (i / sr)))
                    r_val = int(14000 * math.cos(2 * math.pi * 440 * (i / sr)))
                    frames.extend(struct.pack("<hh", l_val, r_val))
                wf.writeframes(frames)

            res = self.restorer.restore_file_to_master_wav(in_wav, out_master)
            self.assertTrue(res["success"])
            self.assertEqual(res["master_sr"], 48000)
            self.assertTrue(out_master.exists())

    def test_pipeline_path_parsing(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            raw_input = f'"{f.name}"  some_invalid_path_xyz.mp3'
            parsed = parse_drag_drop_paths(raw_input)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0].resolve(), Path(f.name).resolve())


    def test_deep_ai_evasion_stages(self):
        from mmm.preserving_sanitizer import AcousticPhysicalRemodelingEngine
        engine = AcousticPhysicalRemodelingEngine(target_sample_rate=48000)
        sr = 44100
        t = np.linspace(0, 0.5, sr // 2)
        audio = np.column_stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 880 * t)])
        
        # 1. Bicoherence phase whitening
        out1 = engine.apply_bicoherence_phase_whitening(audio, sr)
        self.assertEqual(out1.shape, audio.shape)
        self.assertFalse(np.isnan(out1).any())
        
        # 2. Codec frame boundary disruption
        out2 = engine.apply_codec_frame_boundary_disruption(audio, sr)
        self.assertEqual(out2.shape, audio.shape)
        self.assertFalse(np.isnan(out2).any())
        
        # 3. Watermark neutralization
        out3 = engine.neutralize_spectral_watermarks(audio, sr)
        self.assertEqual(out3.shape, audio.shape)
        self.assertFalse(np.isnan(out3).any())
        
        # 4. JND micro-intonation
        out4 = engine.apply_jnd_micro_intonation(audio, sr)
        self.assertEqual(out4.shape, audio.shape)
        self.assertFalse(np.isnan(out4).any())
        
        # 5. HPSS residual whitening
        out5 = engine.apply_hpss_residual_whitening(audio, sr)
        self.assertEqual(out5.shape, audio.shape)
        self.assertFalse(np.isnan(out5).any())
        
        # 6. Full remodel audio
        master, master_sr = engine.remodel_audio(audio, sr)
        self.assertEqual(master_sr, 48000)
        self.assertLessEqual(np.max(np.abs(master)), 0.965)
        self.assertFalse(np.isnan(master).any())

    def test_v3_golden_ear_mono_compatibility_and_forensics(self):
        from mmm.preserving_sanitizer import AcousticPhysicalRemodelingEngine, ForensicCrossValidator
        import mmm.gui as gui_mod

        self.assertTrue(hasattr(gui_mod, "MMMDesktopGUI"))
        self.assertTrue(hasattr(gui_mod, "launch_gui"))

        engine = AcousticPhysicalRemodelingEngine(target_sample_rate=48000)
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        left = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 1760 * t)
        right = 0.4 * np.sin(2 * np.pi * 440 * t + 0.2) + 0.1 * np.cos(2 * np.pi * 1760 * t)
        stereo = np.column_stack([left, right]).astype(np.float32)

        # Verify Zero-Comb Blumlein Stereo guarantees exact mono sum (L_new + R_new == L + R)
        blumlein = engine.apply_stereo_haas_decorrelation(stereo, sr)
        orig_mono = stereo[:, 0] + stereo[:, 1]
        new_mono = blumlein[:, 0] + blumlein[:, 1]
        mono_err = float(np.max(np.abs(orig_mono - new_mono)))
        self.assertLess(mono_err, 1e-5)

        # Verify ForensicCrossValidator 7-domain telemetry
        metrics = ForensicCrossValidator.evaluate_track(blumlein, sr)
        self.assertIn("cepstral_grid_pnr_db", metrics)
        self.assertIn("rvq_periodicity_index", metrics)
        self.assertIn("stereo_hf_correlation", metrics)
        self.assertTrue(metrics["zero_clipping"])


    def test_v3_1_unified_stem_separation_pipeline(self):
        from mmm.master_restoration import UnifiedStemSeparationPipeline, STEM_MODEL_PROFILES

        self.assertIn("bs_roformer_6stem", STEM_MODEL_PROFILES)
        self.assertIn("mel_band_roformer_kim", STEM_MODEL_PROFILES)
        self.assertIn("mvsep_orchestral_fine", STEM_MODEL_PROFILES)

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_wav = Path(tmp_dir) / "mix_input.wav"
            out_stems_dir = Path(tmp_dir) / "Stems_Out"
            sr = 44100
            n = sr // 2
            t = np.linspace(0, 0.5, n, endpoint=False)
            # Synthesize bass (110Hz) + vocal (880Hz) + hi-hat transient
            l_ch = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
            r_ch = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.25 * np.cos(2 * np.pi * 880 * t)
            stereo = np.column_stack([l_ch, r_ch]).astype(np.float32)

            pipe = UnifiedStemSeparationPipeline(target_sr=48000, bit_depth=24)
            pipe._write_wav(in_wav, stereo, sr)

            res = pipe.process_track_to_stems(in_wav, output_dir=out_stems_dir)
            self.assertTrue(res["success"])
            self.assertEqual(len(res["stems"]), 7)
            self.assertIn("vocals", res["stems"])
            self.assertIn("instrumental", res["stems"])
            self.assertTrue(Path(res["resummed_master"]).exists())
            self.assertTrue(res["forensics_after"]["zero_clipping"])

            # Also verify custom 2-stem (Vocals + Instrumental) selection
            pipe_2stem = UnifiedStemSeparationPipeline(
                target_sr=48000,
                bit_depth=24,
                selected_stems=["vocals", "instrumental"],
                export_resummed_master=False,
            )
            res2 = pipe_2stem.process_track_to_stems(in_wav, output_dir=Path(tmp_dir) / "Stems_2Out")
            self.assertEqual(len(res2["stems"]), 2)
            self.assertIsNone(res2["resummed_master"])

            # Verify uv / pip dual package manager auto-detection & Residual Delta Monitor
            mgr, cmd = UnifiedStemSeparationPipeline.detect_package_manager()
            self.assertIn(mgr, ("uv", "pip"))
            self.assertTrue(len(cmd) >= 3)

            pipe_delta = UnifiedStemSeparationPipeline(
                target_sr=48000,
                bit_depth=24,
                selected_stems=["vocals"],
                export_resummed_master=True,
                export_removed_artifact_delta=True,
            )
            res_delta = pipe_delta.process_track_to_stems(in_wav, output_dir=Path(tmp_dir) / "Stems_Delta")
            self.assertTrue(Path(res_delta["removed_artifact_delta"]).exists())
            self.assertIn("rvq_lock", res_delta["stem_metrics"]["vocals"])

            # Verify Stage 16 RIFF BWF/INFO/iXML DAW Container Provenance injection & inspection
            from mmm.acoustic_remodeling import DAWContainerProvenanceEngine, ForensicCrossValidator
            prov = DAWContainerProvenanceEngine.inspect_wav_provenance(Path(res_delta["resummed_master"]))
            self.assertTrue(prov["has_daw_chunks"])
            self.assertIn("bext", prov["chunks"])
            self.assertIn("LIST", prov["chunks"])
            self.assertIn("iXML", prov["chunks"])
            self.assertIn("FL Studio", prov["originator"])


if __name__ == "__main__":
    unittest.main()
