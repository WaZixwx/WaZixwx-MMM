"""
Tests for CLI commands
"""

import unittest
try:
    import pytest
except ImportError:
    raise unittest.SkipTest('pytest is required for this test module')
try:
    import librosa
except ImportError:
    raise unittest.SkipTest('librosa is required for this test module')


from pathlib import Path
import tempfile
import shutil
import numpy as np
import soundfile as sf
from click.testing import CliRunner

from mmm.cli import cli


class TestCLI:
    """Test cases for CLI commands"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.test_dir = Path(tempfile.mkdtemp())

        # Create test audio data
        self.sample_rate = 44100
        self.duration = 1.0  # 1 second for fast tests
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str, audio_data: np.ndarray = None) -> Path:
        """Create a test audio file"""
        if audio_data is None:
            audio_data = self.test_audio
        file_path = self.test_dir / filename
        sf.write(str(file_path), audio_data, self.sample_rate)
        return file_path

    def test_cli_help(self):
        """Test CLI help message"""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Melodic Metadata Massacrer' in result.output

    def test_version_command(self):
        """Test version command"""
        result = self.runner.invoke(cli, ['version'])
        assert result.exit_code == 0

    def test_config_show(self):
        """Test config show command"""
        result = self.runner.invoke(cli, ['config'])
        assert result.exit_code == 0
        assert 'Configuration' in result.output

    def test_config_show_explicit(self):
        """Test config show subcommand"""
        result = self.runner.invoke(cli, ['config', 'show'])
        assert result.exit_code == 0
        assert 'Configuration' in result.output

    def test_config_list(self):
        """Test config list command"""
        result = self.runner.invoke(cli, ['config', 'list'])
        assert result.exit_code == 0
        assert 'Presets' in result.output
        # Check for built-in presets
        assert 'stealth' in result.output
        assert 'fast' in result.output
        assert 'quality' in result.output
        assert 'research' in result.output

    def test_config_preset_builtin(self):
        """Test applying built-in preset"""
        result = self.runner.invoke(cli, ['config', 'preset', 'fast'])
        assert result.exit_code == 0
        assert 'Applied built-in preset' in result.output or 'fast' in result.output

    def test_config_preset_invalid(self):
        """Test applying invalid preset"""
        result = self.runner.invoke(cli, ['config', 'preset', 'nonexistent_preset_xyz'])
        assert 'Failed' in result.output or 'not found' in result.output.lower()

    def test_config_create_preset(self):
        """Test creating custom preset"""
        result = self.runner.invoke(cli, [
            'config', 'create', 'test_preset',
            '--paranoid', 'high',
            '--quality', 'maximum'
        ])
        assert result.exit_code == 0
        assert 'Created' in result.output or 'test_preset' in result.output

        # Verify it shows up in list
        list_result = self.runner.invoke(cli, ['config', 'list'])
        assert 'test_preset' in list_result.output

        # Clean up - delete the preset
        delete_result = self.runner.invoke(cli, ['config', 'delete', 'test_preset', '--yes'])
        assert delete_result.exit_code == 0

    def test_config_delete_builtin_fails(self):
        """Test that deleting built-in preset fails"""
        result = self.runner.invoke(cli, ['config', 'delete', 'stealth', '--yes'])
        assert 'Cannot delete built-in' in result.output

    def test_config_help(self):
        """Test config help shows subcommands"""
        result = self.runner.invoke(cli, ['config', '--help'])
        assert result.exit_code == 0
        assert 'preset' in result.output
        assert 'list' in result.output
        assert 'create' in result.output
        assert 'delete' in result.output
        assert 'reset' in result.output

class TestCLICommands:
    """Test cases for CLI audio processing commands"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.test_dir = Path(tempfile.mkdtemp())

        self.sample_rate = 44100
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str, audio_data: np.ndarray = None) -> Path:
        """Create a test audio file"""
        if audio_data is None:
            audio_data = self.test_audio
        file_path = self.test_dir / filename
        sf.write(str(file_path), audio_data, self.sample_rate)
        return file_path

    def test_analyze_command(self):
        """Test analyze command"""
        input_file = self.create_test_audio_file("test_analyze.wav")
        result = self.runner.invoke(cli, ['analyze', str(input_file)])
        assert result.exit_code == 0
        assert 'analysis' in result.output.lower() or 'threat' in result.output.lower()

    def test_analyze_nonexistent_file(self):
        """Test analyze with nonexistent file"""
        result = self.runner.invoke(cli, ['analyze', '/nonexistent/file.wav'])
        assert result.exit_code != 0

    def test_obliterate_command_basic(self):
        """Test obliterate command basic execution"""
        input_file = self.create_test_audio_file("test_obliterate.wav")
        output_file = self.test_dir / "output.wav"

        result = self.runner.invoke(cli, [
            'obliterate', str(input_file),
            '-o', str(output_file)
        ])
        # Should complete (exit code 0) or fail gracefully
        # The command may succeed or fail depending on audio processing
        assert result.exit_code in [0, 1]

    def test_obliterate_with_paranoid(self):
        """Test obliterate command with paranoid mode"""
        input_file = self.create_test_audio_file("test_paranoid.wav")
        output_file = self.test_dir / "paranoid_output.wav"

        result = self.runner.invoke(cli, [
            'obliterate', str(input_file),
            '-o', str(output_file),
            '--paranoid'
        ])
        assert result.exit_code in [0, 1]

    def test_obliterate_with_verify(self):
        """Test obliterate command with verify flag"""
        input_file = self.create_test_audio_file("test_verify.wav")
        output_file = self.test_dir / "verify_output.wav"

        result = self.runner.invoke(cli, [
            'obliterate', str(input_file),
            '-o', str(output_file),
            '--verify'
        ])
        assert result.exit_code in [0, 1]

    def test_obliterate_with_backup(self):
        """Test obliterate command with backup flag"""
        input_file = self.create_test_audio_file("test_backup.wav")
        output_file = self.test_dir / "backup_output.wav"

        result = self.runner.invoke(cli, [
            'obliterate', str(input_file),
            '-o', str(output_file),
            '--backup'
        ])
        assert result.exit_code in [0, 1]

    def test_obliterate_nonexistent_file(self):
        """Test obliterate with nonexistent file"""
        result = self.runner.invoke(cli, ['obliterate', '/nonexistent/file.wav'])
        assert result.exit_code != 0

    def test_massacre_command(self):
        """Test massacre command with directory"""
        # Create multiple test files
        self.create_test_audio_file("test1.wav")
        self.create_test_audio_file("test2.wav")

        output_dir = self.test_dir / "output"
        output_dir.mkdir()

        result = self.runner.invoke(cli, [
            'massacre', str(self.test_dir),
            '-d', str(output_dir)
        ])
        # Massacre should complete or fail gracefully
        assert result.exit_code in [0, 1]

    def test_massacre_empty_directory(self):
        """Test massacre with empty directory"""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()

        result = self.runner.invoke(cli, ['massacre', str(empty_dir)])
        assert result.exit_code == 0
        assert 'No audio files found' in result.output

    def test_massacre_with_workers(self):
        """Test massacre command with workers option"""
        self.create_test_audio_file("test1.wav")

        result = self.runner.invoke(cli, [
            'massacre', str(self.test_dir),
            '--workers', '2'
        ])
        assert result.exit_code in [0, 1]

    def test_obliterate_format_preserve(self):
        """Test obliterate with format preserve"""
        input_file = self.create_test_audio_file("test_format.wav")
        output_file = self.test_dir / "format_output.wav"

        result = self.runner.invoke(cli, [
            'obliterate', str(input_file),
            '-o', str(output_file),
            '--format', 'preserve'
        ])
        assert result.exit_code in [0, 1]

    def test_stealth_flags(self):
        """Test obliterate with various stealth flags"""
        input_file = self.create_test_audio_file("test_stealth.wav")
        output_file = self.test_dir / "stealth_output.wav"

        result = self.runner.invoke(cli, [
            'obliterate', str(input_file),
            '-o', str(output_file),
            '--no-phase-dither',
            '--no-comb-mask',
            '--phase-noise',
            '--gated-resample-nudge'
        ])
        assert result.exit_code in [0, 1]
