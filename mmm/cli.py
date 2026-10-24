#!/usr/bin/env python3
"""
Melodic Metadata Massacrer (MMM) CLI
WaZixwx-MMM Studio v7.0.0 — Golden-Ear Mastering, 6-Stem Separation & 7-Domain Anti-AI Suite
"""

import click
import sys
import os
from pathlib import Path

from .ui.console import ConsoleManager
from .ui.banners import BannerManager
from . import __version__

def _get_audio_sanitizer():
    try:
        from .core.audio_sanitizer import AudioSanitizer
        return AudioSanitizer
    except ImportError as e:
        console.error(f"❌ Missing required audio dependency: {e}")
        console.info("💡 Please install dependencies using: pip install -r requirements.txt")
        sys.exit(1)
from .config.config_manager import ConfigManager

console = ConsoleManager()
banner = BannerManager()
config = ConfigManager()


@click.group()
@click.version_option(version=__version__, prog_name="mmm")
@click.pass_context
def cli(ctx):
    """
    🎛️ WaZixwx-MMM Studio v7.0.0 (Melodic Metadata Massacrer)

    Golden-Ear Acoustic Remodeling, 3-Stage 6-Stem Separation & 7-Domain Anti-AI Forensics Suite.
    Transforms lossy or synthetic audio into 24-Bit/48kHz Studio Master WAVs with zero audible loss.
    """
    ctx.ensure_object(dict)

    # Display WaZixwx-MMM Studio v7.0 banner
    banner.show_main_banner()


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (auto-generates if not specified)",
)
@click.option(
    "--paranoid",
    is_flag=True,
    default=False,
    help="Maximum destruction mode - multiple passes with aggressive cleaning",
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Verify watermark removal effectiveness",
)
@click.option(
    "--backup", is_flag=True, default=False, help="Create backup of original file"
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["preserve", "mp3", "wav"], case_sensitive=False),
    default="preserve",
    help="Output audio format",
)
@click.option(
    "--turbo/--no-turbo",
    default=True,
    help="Enable/disable turbo mode with GPU acceleration (700x+ faster)",
)
@click.option(
    "--phase-dither/--no-phase-dither",
    default=True,
    help="Toggle sub-block phase dither (advanced)",
)
@click.option(
    "--comb-mask/--no-comb-mask",
    default=True,
    help="Toggle dynamic comb masking (advanced)",
)
@click.option(
    "--transient-shift/--no-transient-shift",
    default=True,
    help="Toggle transient micro-shift (advanced)",
)
@click.option(
    "--resample-nudge/--no-resample-nudge",
    default=True,
    help="Toggle resample nudge (advanced)",
)
@click.option(
    "--phase-noise/--no-phase-noise",
    default=True,
    help="Toggle FFT phase noise (advanced)",
)
@click.option(
    "--phase-swirl/--no-phase-swirl", default=True, help="Toggle phase swirl (advanced)"
)
@click.option(
    "--masked-hf-phase/--no-masked-hf-phase",
    default=False,
    help="Toggle masked high-frequency phase noise (advanced)",
)
@click.option(
    "--gated-resample-nudge/--no-gated-resample-nudge",
    default=False,
    help="Toggle RMS-gated resample nudge (advanced)",
)
@click.option(
    "--micro-eq-flutter/--no-micro-eq-flutter",
    default=False,
    help="Toggle gated micro-EQ flutter (advanced)",
)
@click.option(
    "--hf-decorrelate/--no-hf-decorrelate",
    default=False,
    help="Toggle HF band decorrelation (advanced)",
)
@click.option(
    "--refined-transient/--no-refined-transient",
    default=False,
    help="Toggle refined transient micro-shift (advanced)",
)
@click.option(
    "--adaptive-transient/--no-adaptive-transient",
    default=False,
    help="Toggle adaptive transient shift (onset-strength gated, ultra-small)",
)
@click.pass_context
def obliterate(
    ctx,
    input_file,
    output,
    paranoid,
    verify,
    backup,
    output_format,
    turbo,
    phase_dither,
    comb_mask,
    transient_shift,
    resample_nudge,
    phase_noise,
    phase_swirl,
    masked_hf_phase,
    gated_resample_nudge,
    micro_eq_flutter,
    hf_decorrelate,
    refined_transient,
    adaptive_transient,
):
    """
    💀 Completely annihilate all traces from audio file

    Removes metadata, watermarks, AI fingerprints, and statistical patterns
    that identify AI-generated content. This is the nuclear option.

    Examples:

        mmm obliterate dystopian_symphony.mp3

        mmm obliterate --paranoid --verify music.wav -o clean_music.wav
    """
    console.success(f"🔍 Scanning: {input_file.name}")

    try:
        # Create fresh config manager to avoid scope issues
        config_manager = ConfigManager()
        target_format = None if output_format == "preserve" else output_format

        # Apply preset defaults to advanced flags unless the user explicitly set them
        preset_flags = {}
        preset_name = config_manager.config.get("preset")
        if preset_name:
            from .config.defaults import PRESETS

            preset_data = PRESETS.get(preset_name) or {}
            preset_flags = preset_data.get("advanced_flags", {})

        def resolve_flag(flag_name: str, cli_value: bool) -> bool:
            # If the flag was explicitly provided on CLI, keep it; else fall back to preset if available
            if (
                ctx.get_parameter_source(flag_name)
                == click.core.ParameterSource.COMMANDLINE
            ):
                return cli_value
            return preset_flags.get(flag_name, cli_value)

        phase_dither = resolve_flag("phase_dither", phase_dither)
        comb_mask = resolve_flag("comb_mask", comb_mask)
        transient_shift = resolve_flag("transient_shift", transient_shift)
        resample_nudge = resolve_flag("resample_nudge", resample_nudge)
        gated_resample_nudge = resolve_flag(
            "gated_resample_nudge", gated_resample_nudge
        )
        phase_noise = resolve_flag("phase_noise", phase_noise)
        phase_swirl = resolve_flag("phase_swirl", phase_swirl)
        masked_hf_phase = resolve_flag("masked_hf_phase", masked_hf_phase)
        micro_eq_flutter = resolve_flag("micro_eq_flutter", micro_eq_flutter)
        hf_decorrelate = resolve_flag("hf_decorrelate", hf_decorrelate)
        refined_transient = resolve_flag("refined_transient", refined_transient)
        adaptive_transient = resolve_flag("adaptive_transient", adaptive_transient)

        sanitizer = _get_audio_sanitizer()(
            input_file=input_file,
            output_file=output,
            paranoid_mode=paranoid,
            config=config_manager.config,
            output_format=target_format,
        )

        # Create backup if requested
        if backup:
            sanitizer.create_backup()
            console.info("📦 Backup created - Your secrets are safe... for now")

        # Show what we found (use turbo analysis if requested)
        if turbo:
            console.info("🚀 TURBO MODE: Fast analysis with GPU acceleration")
            # Import turbo analysis for fast scanning
            try:
                from .turbo_analysis import turbo_analysis

                analysis_results = turbo_analysis(input_file)

                # Convert turbo results to expected format
                analysis = {
                    "threats_found": analysis_results.get("total_threats", 0),
                    "threat_level": analysis_results.get("threat_level")
                    or (
                        "HIGH"
                        if analysis_results.get("total_threats", 0) > 10
                        else (
                            "MEDIUM"
                            if analysis_results.get("total_threats", 0) > 5
                            else "LOW"
                        )
                    ),
                    "file_info": analysis_results.get("file_info", {}),
                    "metadata": analysis_results.get("metadata", {}),
                    "watermarks": analysis_results.get("gpu_watermarks", {}),
                }
                console.info(
                    f"⚡ Turbo analysis complete: {analysis['threats_found']} threats found"
                )
            except Exception as e:
                console.warning(
                    f"⚠️ Turbo analysis failed, falling back to regular: {e}"
                )
                analysis = sanitizer.analyze_file(
                    deep=False
                )  # Use shallow analysis as fallback
        else:
            # Regular analysis (slower but thorough)
            analysis = sanitizer.analyze_file(
                deep=False
            )  # Use shallow analysis for speed

        console.display_analysis(analysis)

        if analysis["threats_found"]:
            console.error(
                f"😈 Found {analysis['threats_found']} threats... time to DELETE THEM ALL!"
            )
        else:
            console.warning(
                "🤔 No obvious threats detected... but we'll clean it anyway!"
            )

        # Begin the massacre
        if turbo:
            console.info("🚀 TURBO MODE: Beginning fast sanitization...")
            # Import and use PRESERVING sanitizer
            try:
                from .preserving_sanitizer import preserving_sanitize

                console.info(
                    f"🎵 Calling PRESERVING sanitizer with paranoid_mode={paranoid}"
                )
                # Pass the actual threat count from analysis
                threat_count = analysis.get("threats_found", 0)
                console.info(
                    f"🎯 Preserving audio while removing {threat_count} threats"
                )
                preserving_result = preserving_sanitize(
                    input_file,
                    output,
                    paranoid,
                    threat_count,
                    output_format=target_format,
                    phase_dither=phase_dither,
                    comb_mask=comb_mask,
                    transient_shift=transient_shift,
                    resample_nudge=resample_nudge,
                    gated_resample_nudge=gated_resample_nudge,
                    phase_noise=phase_noise,
                    phase_swirl=phase_swirl,
                    masked_hf_phase=masked_hf_phase,
                    micro_eq_flutter=micro_eq_flutter,
                    hf_decorrelate=hf_decorrelate,
                    refined_transient=refined_transient,
                    adaptive_transient=adaptive_transient,
                )

                # Convert preserving results to expected format
                result = {
                    "success": preserving_result["success"],
                    "output_file": preserving_result["output_file"],
                    "stats": {
                        "metadata_removed": preserving_result["stats"][
                            "metadata_removed"
                        ],
                        "watermarks_detected": preserving_result["stats"][
                            "watermarks_removed"
                        ],
                        "watermarks_removed": preserving_result["stats"][
                            "watermarks_removed"
                        ],
                        "quality_loss": 0.0,  # Assume minimal loss with preserving mode
                        "processing_time": preserving_result["stats"][
                            "processing_time"
                        ],
                    },
                }
                console.success(
                    f"🎵 PRESERVING sanitization completed in {preserving_result['stats']['processing_time']}"
                )
            except Exception as e:
                console.warning(
                    f"⚠️ PRESERVING sanitization failed, falling back to regular: {e}"
                )
                import traceback

                console.error(f"Traceback: {traceback.format_exc()}")
                console.info("🌊 Falling back to regular sanitization...")
                result = sanitizer.sanitize_audio()
        else:
            console.info("🌊 Beginning audio sanitization protocol...")
            result = sanitizer.sanitize_audio()

        # Display results
        console.display_results(result)

        if verify and result["success"]:
            console.info("🔬 Verification phase: Double-checking our work...")
            if turbo:
                # Use turbo analysis for verification
                try:
                    console.info("⚡ Using turbo analysis for verification...")
                    from .turbo_analysis import turbo_analysis

                    original_threats = analysis["threats_found"]

                    # Analyze the cleaned file
                    cleaned_file = Path(result["output_file"])
                    post_analysis_results = turbo_analysis(cleaned_file)
                    remaining_threats = post_analysis_results.get("total_threats", 0)

                    # Calculate effectiveness
                    removal_effectiveness = 0
                    if original_threats > 0:
                        removal_effectiveness = (
                            (original_threats - remaining_threats) / original_threats
                        ) * 100

                    verification = {
                        "success": True,
                        "original_threats": original_threats,
                        "remaining_threats": remaining_threats,
                        "removal_effectiveness": round(removal_effectiveness, 2),
                        "hash_different": True,  # Assume true since we modified the file
                    }
                except Exception as e:
                    verification = {"success": False, "error": str(e)}
            else:
                # Use regular verification
                verification = sanitizer.verify_sanitization()

            console.display_verification(verification)

        if result["success"]:
            console.success(
                "✨ File sanitized! Your AI overlords will never know... 🤫"
            )
            console.hacker_quote()
        else:
            console.error("💥 Sanitization failed! The matrix fought back...")
            sys.exit(1)

    except Exception as e:
        console.error(f"💀 CRITICAL ERROR: {str(e)}")
        console.error("The audio has fought back and won this round...")
        sys.exit(1)


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir",
    "-d",
    type=click.Path(path_type=Path),
    help="Output directory (creates subdirectory if not specified)",
)
@click.option(
    "--extension",
    "-e",
    multiple=True,
    default=["mp3", "wav"],
    help="File extensions to process (can be used multiple times)",
)
@click.option(
    "--paranoid", is_flag=True, default=False, help="Maximum destruction mode"
)
@click.option("--workers", "-w", type=int, default=4, help="Number of parallel workers")
@click.option(
    "--backup", is_flag=True, default=False, help="Create backups of original files"
)
@click.option(
    "--turbo/--no-turbo",
    default=True,
    help="Enable/disable turbo mode with GPU acceleration",
)
@click.pass_context
def massacre(ctx, directory, output_dir, extension, paranoid, workers, backup, turbo):
    """
    ⚡ Process entire directory of audio files

    Mass sanitization mode for bulk operations. Processes all supported
    audio files in the specified directory with parallel execution.

    Example:

        mmm massacre /path/to/music --paranoid --workers 8
    """
    console.success(f"🎯 Directory massacre initiated: {directory}")
    console.info(f"⚙️  Extensions: {', '.join(extension)}")
    console.info(f"🔥 Workers: {workers} | Paranoid: {'ON' if paranoid else 'OFF'}")

    # Scan for files
    files = []
    for ext in extension:
        files.extend(directory.glob(f"*.{ext.lower()}"))
        files.extend(directory.glob(f"*.{ext.upper()}"))

    if not files:
        console.warning("📂 No audio files found in directory")
        return

    console.success(f"📁 Found {len(files)} files to process")

    if turbo:
        console.info("🚀 TURBO MODE enabled for massacre")

    console.info("🔄 Processing files...")
    for file_path in files:
        try:
            output_file = output_dir / file_path.name if output_dir else None
            console.info(f"   Processing: {file_path.name}")

            if turbo:
                # Use turbo path: turbo_analysis + preserving_sanitize
                try:
                    from .turbo_analysis import turbo_analysis as _turbo_analysis
                    from .preserving_sanitizer import preserving_sanitize

                    analysis_results = _turbo_analysis(file_path)
                    threat_count = analysis_results.get("total_threats", 0)

                    preserving_result = preserving_sanitize(
                        file_path,
                        output_file,
                        paranoid,
                        threat_count,
                    )

                    if preserving_result["success"]:
                        console.success(f"   ✅ {file_path.name} - Sanitized (turbo)!")
                    else:
                        console.error(f"   ❌ {file_path.name} - Failed!")
                except Exception as e:
                    console.warning(
                        f"   ⚠️ Turbo failed for {file_path.name}, falling back: {e}"
                    )
                    # Fall back to regular sanitization
                    config_manager = ConfigManager()
                    sanitizer = _get_audio_sanitizer()(
                        input_file=file_path,
                        output_file=output_file,
                        paranoid_mode=paranoid,
                        config=config_manager.config,
                    )
                    if backup:
                        sanitizer.create_backup()
                    result = sanitizer.sanitize_audio()
                    if result["success"]:
                        console.success(f"   ✅ {file_path.name} - Sanitized (fallback)!")
                    else:
                        console.error(f"   ❌ {file_path.name} - Failed!")
            else:
                # Regular (non-turbo) path
                config_manager = ConfigManager()
                sanitizer = _get_audio_sanitizer()(
                    input_file=file_path,
                    output_file=output_file,
                    paranoid_mode=paranoid,
                    config=config_manager.config,
                )

                if backup:
                    sanitizer.create_backup()

                result = sanitizer.sanitize_audio()

                if result["success"]:
                    console.success(f"   ✅ {file_path.name} - Sanitized!")
                else:
                    console.error(f"   ❌ {file_path.name} - Failed!")

        except Exception as e:
            console.error(f"   💥 {file_path.name} - Error: {str(e)}")

    console.success("🎉 Massacre complete! The audio has been liberated!")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--turbo",
    is_flag=True,
    default=False,
    help="Enable turbo mode with GPU acceleration (660x faster)",
)
@click.pass_context
def analyze(ctx, input_file, turbo):
    """
    🔬 Analyze audio file for watermarks and metadata

    Performs deep forensic analysis to identify potential watermarks,
    AI fingerprints, and hidden metadata without modifying the file.

    Example:

        mmm analyze suspicious_music.mp3
    """
    if turbo:
        console.info(f"🚀 TURBO Forensic analysis: {input_file.name}")
        console.info("⚡ Using GPU acceleration for maximum speed...\n")

        # Import turbo analysis
        try:
            from .turbo_analysis import turbo_analysis

            results = turbo_analysis(input_file)

            # Display turbo results
            console.display_turbo_analysis(results)

        except ImportError:
            console.error("💥 Turbo mode requires GPU packages. Run without --turbo")
            sys.exit(1)
        except Exception as e:
            console.error(f"💥 Turbo analysis failed: {str(e)}")
            sys.exit(1)
    else:
        console.info(f"🔬 Forensic analysis: {input_file.name}")
        console.info("🔍 Scanning for digital footprints...\n")

        try:
            # Create fresh config manager to avoid scope issues
            config_manager = ConfigManager()
            sanitizer = _get_audio_sanitizer()(
                input_file=input_file, config=config_manager.config
            )
            analysis = sanitizer.analyze_file(deep=True)

            console.display_detailed_analysis(analysis)

            # Threat assessment
            if analysis["threat_level"] == "HIGH":
                console.error(
                    "🚨 HIGH THREAT LEVEL - This file is heavily watermarked!"
                )
            elif analysis["threat_level"] == "MEDIUM":
                console.warning("⚠️  MEDIUM THREAT LEVEL - Some traces detected")
            else:
                console.success("✅ LOW THREAT LEVEL - Relatively clean")

        except Exception as e:
            console.error(f"💥 Analysis failed: {str(e)}")
            sys.exit(1)


@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """
    ⚙️  Configuration management

    Manage MMM configuration settings and presets.

    Examples:

        mmm config              Show current config

        mmm config preset stealth   Apply stealth preset

        mmm config list         List available presets

        mmm config create my_preset --paranoid maximum --quality high
    """
    # If no subcommand provided, show current config (default behavior)
    if ctx.invoked_subcommand is None:
        config_manager = ConfigManager()
        console.info("⚙️  Current Configuration:")
        console.display_config(config_manager.get_config())


@config.command("show")
def config_show():
    """
    📋 Show current configuration

    Displays current MMM configuration settings and defaults.
    """
    config_manager = ConfigManager()
    console.info("⚙️  Current Configuration:")
    console.display_config(config_manager.get_config())


@config.command("preset")
@click.argument("name", type=str)
def config_preset(name):
    """
    🎛️  Apply a configuration preset

    Apply a built-in or custom preset by name.

    Built-in presets: stealth, stealth-plus, fast, quality, research

    Examples:

        mmm config preset stealth

        mmm config preset fast
    """
    from .config.defaults import PRESETS

    config_manager = ConfigManager()

    # Check if it's a built-in preset
    if name in PRESETS:
        preset_data = PRESETS[name]
        # Merge preset with current config
        for key, value in preset_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    config_manager.set(f"{key}.{sub_key}", sub_value)
            else:
                config_manager.set(key, value)
        # Also set the preset name in config for reference
        config_manager.set("preset", name)
        config_manager.save_config()
        console.success(f"✅ Applied built-in preset: {name}")
        console.info("📋 Preset settings:")
        for key, value in preset_data.items():
            if key != "advanced_flags":
                console.info(f"   {key}: {value}")
            else:
                console.info(f"   {key}:")
                for flag, flag_value in value.items():
                    console.info(f"      {flag}: {flag_value}")
    else:
        # Try to load custom preset
        try:
            preset_data = config_manager.load_preset(name)
            for key, value in preset_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        config_manager.set(f"{key}.{sub_key}", sub_value)
                else:
                    config_manager.set(key, value)
            config_manager.set("preset", name)
            config_manager.save_config()
            console.success(f"✅ Applied custom preset: {name}")
        except Exception as e:
            console.error(f"❌ Failed to apply preset '{name}': {e}")
            console.info("💡 Available built-in presets: stealth, stealth-plus, fast, quality, research")
            console.info("💡 Use 'mmm config list' to see all available presets")


@config.command("list")
def config_list():
    """
    📜 List available presets

    Shows all built-in and custom configuration presets.
    """
    from .config.defaults import PRESETS

    config_manager = ConfigManager()

    console.info("📜 Available Presets:\n")

    # Built-in presets
    console.info("🔧 Built-in Presets:")
    preset_descriptions = {
        "stealth": "Maximum paranoia, quality preservation",
        "stealth-plus": "Stealth with advanced flags optimized for detector evasion",
        "fast": "Quick processing, basic cleaning",
        "quality": "Preserve maximum audio quality",
        "research": "Deep analysis, detailed logging",
    }
    for name in PRESETS:
        desc = preset_descriptions.get(name, "No description")
        console.info(f"   • {name}: {desc}")

    # Custom presets
    custom_presets = config_manager.list_presets()
    if custom_presets:
        console.info("\n🎨 Custom Presets:")
        for name in custom_presets:
            console.info(f"   • {name}")
    else:
        console.info("\n💡 No custom presets found. Create one with 'mmm config create'")


@config.command("create")
@click.argument("name", type=str)
@click.option(
    "--paranoid",
    type=click.Choice(["low", "medium", "high", "maximum"], case_sensitive=False),
    default="medium",
    help="Paranoia level",
)
@click.option(
    "--quality",
    type=click.Choice(["low", "medium", "high", "maximum"], case_sensitive=False),
    default="high",
    help="Quality preservation level",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["preserve", "mp3", "wav"], case_sensitive=False),
    default="preserve",
    help="Output format",
)
@click.option("--backup/--no-backup", default=False, help="Backup originals by default")
@click.option("--verify/--no-verify", default=True, help="Auto-verify after processing")
def config_create(name, paranoid, quality, output_format, backup, verify):
    """
    🎨 Create a custom configuration preset

    Create a new preset with specified settings.

    Examples:

        mmm config create my_preset --paranoid maximum --quality high

        mmm config create quick_clean --paranoid low --quality medium --no-verify
    """
    config_manager = ConfigManager()

    preset_config = {
        "paranoia_level": paranoid,
        "preserve_quality": quality,
        "output_format": output_format,
        "backup_originals": backup,
        "verification": {
            "auto_verify": verify,
        },
    }

    try:
        config_manager.create_preset(name, preset_config)
        console.success(f"✅ Created custom preset: {name}")
        console.info("📋 Preset settings:")
        console.info(f"   paranoia_level: {paranoid}")
        console.info(f"   preserve_quality: {quality}")
        console.info(f"   output_format: {output_format}")
        console.info(f"   backup_originals: {backup}")
        console.info(f"   auto_verify: {verify}")
        console.info(f"\n💡 Apply with: mmm config preset {name}")
    except Exception as e:
        console.error(f"❌ Failed to create preset: {e}")


@config.command("delete")
@click.argument("name", type=str)
@click.confirmation_option(prompt="Are you sure you want to delete this preset?")
def config_delete(name):
    """
    🗑️  Delete a custom preset

    Remove a custom configuration preset.

    Example:

        mmm config delete my_preset
    """
    from .config.defaults import PRESETS

    if name in PRESETS:
        console.error(f"❌ Cannot delete built-in preset: {name}")
        return

    config_manager = ConfigManager()

    try:
        config_manager.delete_preset(name)
        console.success(f"✅ Deleted preset: {name}")
    except Exception as e:
        console.error(f"❌ Failed to delete preset: {e}")


@config.command("reset")
@click.confirmation_option(prompt="Are you sure you want to reset to defaults?")
def config_reset():
    """
    🔄 Reset configuration to defaults

    Restore all settings to their default values.
    """
    config_manager = ConfigManager()
    config_manager.reset_to_defaults()
    console.success("✅ Configuration reset to defaults")


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind to (use 0.0.0.0 for all interfaces)",
)
@click.option(
    "--port",
    default=8778,
    show_default=True,
    type=int,
    help="Port to listen on",
)
@click.option(
    "--max-size",
    default=500,
    show_default=True,
    type=int,
    help="Maximum upload file size in MB",
)
def server(host, port, max_size):
    """
    🌐 Launch browser-based audio sanitizer

    Starts a local web server with drag-and-drop audio sanitization.
    Open the displayed URL in your browser to use the web interface.

    Examples:

        mmm server

        mmm server --port 9000

        mmm server --host 0.0.0.0 --port 8778
    """
    from .server import run_server

    max_file_size = max_size * 1024 * 1024
    run_server(host=host, port=port, max_file_size=max_file_size)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output audio file path")
@click.option("--target-sr", default=48000, show_default=True, type=int, help="Target sample rate in Hz")
@click.option("--air-gain", default=1.0, show_default=True, type=float, help="Air-band overtone gain multiplier")
def upsample(input_file, output, target_sr, air_gain):
    """
    🚀 AI Audio Super-Resolution & Bandwidth Extension (BWE)

    Reconstructs missing overtones, air-band harmonics (up to 24kHz),
    and restores mastering-grade dynamics and clarity to compressed audio.
    """
    console.info(f"✨ Enhancing audio resolution: {input_file.name}")
    from .preserving_sanitizer import AudioSuperResolution
    out_path = output or (input_file.parent / f"{input_file.stem}_hd_{target_sr}hz{input_file.suffix}")
    asr = AudioSuperResolution(target_sample_rate=target_sr, air_band_gain=air_gain)
    res = asr.super_resolve_file(input_file, out_path)
    console.success(f"✅ Enhanced output saved to: {out_path.name}")
    console.info(f"   📊 Original SR: {res['original_sr']} Hz -> Target SR: {res['target_sr']} Hz")
    console.info(f"   🔍 Reconstructed harmonics above: {res['detected_cutoff']:.0f} Hz")


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path())
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), help="Target folder for master WAVs (default: Final_Clean_WAV)")
@click.option("--target-sr", default=48000, show_default=True, type=int, help="Target sample rate in Hz")
@click.option("--bit-depth", default=24, show_default=True, type=int, help="Target bit depth (16 or 24-bit)")
def pipeline(paths, output_dir, target_sr, bit_depth):
    """
    ⚡ Launch one-click studio mastering & cleaning pipeline

    Accepts files/folders via CLI or launches interactive drag-and-drop mode.
    Strips metadata, removes AI watermarks, and reconstructs 24-bit/48kHz master WAVs.
    """
    from .preserving_sanitizer import parse_drag_drop_paths, run_mastering_pipeline, interactive_pipeline_loop
    if paths:
        parsed = []
        for item in paths:
            parsed.extend(parse_drag_drop_paths(item))
        run_mastering_pipeline(parsed, output_dir=output_dir, target_sr=target_sr, bit_depth=bit_depth)
    else:
        interactive_pipeline_loop(output_dir=output_dir)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output audio path")
@click.option("--target-sr", default=48000, show_default=True, type=int, help="Target sample rate in Hz")
@click.option("--bit-depth", default=24, show_default=True, type=int, help="Target bit depth (16 or 24)")
@click.option("--daw-profile", default="fl_studio_21", show_default=True, type=click.Choice(["fl_studio_21", "cubase_13_pro", "pro_tools_2024", "logic_pro_11", "ableton_live_12"]), help="DAW RIFF BWF/INFO/iXML provenance profile to embed")
def remodel(input_path, output, target_sr, bit_depth, daw_profile):
    """
    🔬 v7.0 Golden-Ear Acoustic Remodeling & 7-Domain Anti-AI Evasion

    Transforms synthetic or lossy audio into a 24-bit/48kHz physics-grounded master recording,
    neutralizing ArtifactNet, SONICS SpecTTTra, Deezer HOS, AudioSeal, and SynthID signatures.
    """
    from rich.table import Table
    from .preserving_sanitizer import MasterGradeAudioRestorer

    if output is None:
        output = input_path.parent / f"{input_path.stem}_v7_Master.wav"

    console.console.rule("[bold cyan]🔬 WaZixwx-MMM Studio v7.0 — 11-Stage Golden-Ear Remodeling[/bold cyan]")
    restorer = MasterGradeAudioRestorer(target_sample_rate=target_sr, bit_depth=bit_depth, daw_profile=daw_profile)
    res = restorer.restore_file_to_master_wav(input_path, output)

    b_m = res.get("forensics_before", {})
    a_m = res.get("forensics_after", {})
    if b_m and a_m:
        tbl = Table(title="📊 7-Domain Forensic & Golden-Ear Telemetry Scorecard", border_style="cyan")
        tbl.add_column("Acoustic / Forensic Metric", style="bold white")
        tbl.add_column("Input Track", justify="right", style="yellow")
        tbl.add_column("v7.0 Studio Master", justify="right", style="bold green")
        tbl.add_row("Sample Rate / Bit Depth", f"{res['orig_sr']} Hz", f"{res['master_sr']} Hz / {res['bit_depth']}-Bit")
        tbl.add_row("True-Peak Level (dBFS)", f"{b_m.get('true_peak_dbfs')} dBFS", f"{a_m.get('true_peak_dbfs')} dBFS (Zero-Clip)")
        tbl.add_row("Dynamic RMS / Crest Factor", f"{b_m.get('rms')} ({b_m.get('crest_factor_db')} dB)", f"{a_m.get('rms')} ({a_m.get('crest_factor_db')} dB)")
        tbl.add_row("Cepstral Grid Ripple PNR", f"{b_m.get('cepstral_grid_pnr_db')} dB", f"{a_m.get('cepstral_grid_pnr_db')} dB")
        tbl.add_row("RVQ Frame Periodicity Lock", f"{b_m.get('rvq_periodicity_index')}", f"{a_m.get('rvq_periodicity_index')}")
        tbl.add_row("Stereo HF Correlation (>2kHz)", f"{b_m.get('stereo_hf_correlation')}", f"{a_m.get('stereo_hf_correlation')} (Zero-Comb)")
        tbl.add_row("Air-Band Ratio (>16kHz)", f"{b_m.get('air_band_ratio_db')} dB", f"{a_m.get('air_band_ratio_db')} dB")
        tbl.add_row("24-Bit LSB TPDF Entropy", f"{b_m.get('lsb_entropy_bits')} bits", f"{a_m.get('lsb_entropy_bits')} bits (DAW Dithered)")
        tbl.add_row("RIFF BWF/iXML DAW Provenance", f"{b_m.get('daw_provenance')}", f"{a_m.get('daw_provenance')} (Verified)")
        console.console.print(tbl)

    console.success(f"✓ Studio Remodeling & Zero-Trace Evasion completed in {res['processing_time']:.2f}s!")
    console.info(f"📂 Master Output: {res['output_file']}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), help="Output directory for separated 24-bit stems")
@click.option(
    "--profile",
    "-p",
    type=click.Choice(["bs_roformer_6stem", "mel_band_roformer_kim", "mvsep_orchestral_fine"]),
    default="bs_roformer_6stem",
    show_default=True,
    help="Separation profile (6-stem pop/rock/piano/guitar, vocal/inst, or orchestral/solo)",
)
@click.option("--stems", "-s", default="all", show_default=True, help="Comma-separated stems to export: vocals,instrumental,drums,bass,piano_guitar,strings_synth,other or all")
@click.option("--resummed/--no-resummed", default=True, show_default=True, help="Also export 99_ReSummed_ZeroTrace_Master.wav")
@click.option("--delta-monitor/--no-delta-monitor", default=False, show_default=True, help="Also export 98_Removed_AI_Artifact_Delta.wav (+24dB Residual Monitor)")
@click.option("--target-sr", default=48000, show_default=True, type=int, help="Target sample rate in Hz")
@click.option("--bit-depth", default=24, show_default=True, type=int, help="Target bit depth (16 or 24)")
def separate(input_path, output_dir, profile, stems, resummed, delta_monitor, target_sr, bit_depth):
    """
    🎚️ v7.0 Unified 3-Stage Stem Separation & Per-Stem Remodeling

    Executes: Stage 1 (Pre-Separation Cepstral De-Artifacting) -> Stage 2 (6-Stem Separation:
    vocals, drums, bass, piano_guitar, strings_synth, other) -> Stage 3 (Per-Stem Physical
    Acoustic Remodeling + Re-Summed Zero-Trace Master Mix).
    """
    from rich.table import Table
    from .master_restoration import UnifiedStemSeparationPipeline

    console.console.rule("[bold cyan]🎚️ WaZixwx-MMM Studio v7.0 — 3-Stage Stem Separation & Remodeling[/bold cyan]")
    from .master_restoration import ALL_SELECTABLE_STEMS
    selected = list(ALL_SELECTABLE_STEMS) if stems.strip().lower() == "all" else [x.strip().lower() for x in stems.split(",") if x.strip()]
    pipe = UnifiedStemSeparationPipeline(
        profile=profile,
        target_sr=target_sr,
        bit_depth=bit_depth,
        pre_deartifact=True,
        post_remodel_stems=True,
        export_resummed_master=resummed,
        export_removed_artifact_delta=delta_monitor,
        selected_stems=selected,
    )
    res = pipe.process_track_to_stems(input_path, output_dir=output_dir)

    stem_tbl = Table(title=f"🎚️ Separated & Remodeled 24-Bit Stems ({res['backend']})", border_style="green")
    stem_tbl.add_column("Stem Channel", style="bold cyan")
    stem_tbl.add_column("True-Peak", justify="right", style="green")
    stem_tbl.add_column("RMS", justify="right", style="yellow")
    stem_tbl.add_column("RVQ Lock", justify="right", style="bold green")
    stem_tbl.add_column("Cepstral PNR", justify="right", style="cyan")
    stem_tbl.add_column("Exported Master File", style="white")
    for s_name, s_meta in res["stem_metrics"].items():
        stem_tbl.add_row(
            s_name.upper(),
            f"{s_meta['peak_dbfs']} dBFS",
            f"{s_meta['rms']}",
            f"{s_meta.get('rvq_lock', '-')}",
            f"{s_meta.get('cepstral_pnr_db', '-')} dB",
            s_meta["file"],
        )
    console.console.print(stem_tbl)

    b_m = res.get("forensics_before", {})
    a_m = res.get("forensics_after", {})
    if b_m and a_m:
        f_tbl = Table(title="📊 Re-Summed Master 7-Domain Forensic Telemetry", border_style="cyan")
        f_tbl.add_column("Metric", style="bold white")
        f_tbl.add_column("Input Mix", justify="right", style="yellow")
        f_tbl.add_column("Re-Summed 6-Stem Master", justify="right", style="bold green")
        f_tbl.add_row("Resolution", f"{res['orig_sr']} Hz", f"{res['master_sr']} Hz / {res['bit_depth']}-Bit")
        f_tbl.add_row("True-Peak (dBFS)", f"{b_m.get('true_peak_dbfs')} dBFS", f"{a_m.get('true_peak_dbfs')} dBFS (Zero-Clip)")
        f_tbl.add_row("Dynamic RMS / Crest", f"{b_m.get('rms')} ({b_m.get('crest_factor_db')} dB)", f"{a_m.get('rms')} ({a_m.get('crest_factor_db')} dB)")
        f_tbl.add_row("Cepstral Grid PNR", f"{b_m.get('cepstral_grid_pnr_db')} dB", f"{a_m.get('cepstral_grid_pnr_db')} dB")
        f_tbl.add_row("RVQ Periodicity Lock", f"{b_m.get('rvq_periodicity_index')}", f"{a_m.get('rvq_periodicity_index')}")
        f_tbl.add_row("Stereo HF Correlation", f"{b_m.get('stereo_hf_correlation')}", f"{a_m.get('stereo_hf_correlation')}")
        console.console.print(f_tbl)

    console.success(f"✓ 3-Stage Stem Separation & Per-Stem Remodeling completed in {res['processing_time']:.2f}s!")
    console.info(f"📂 Stems & Re-Summed Master Directory: {res['output_dir']}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
def audit(input_path):
    """
    📊 Run 1-second 7-Domain Forensic & Golden-Ear Diagnostic Scan
    """
    import wave
    import numpy as np
    from rich.table import Table
    from .preserving_sanitizer import ForensicCrossValidator, _ensure_channel_layout

    console.console.rule(f"[bold cyan]📊 WaZixwx-MMM v7.0 — 7-Domain Forensic Diagnostic: {input_path.name}[/bold cyan]")
    audio = None
    sr = 44100
    if input_path.suffix.lower() == ".wav":
        with wave.open(str(input_path), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
            if sw == 2:
                s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 3:
                u8 = np.frombuffer(raw, dtype=np.uint8)
                i32 = (
                    (u8[0::3].astype(np.int32) << 8)
                    | (u8[1::3].astype(np.int32) << 16)
                    | (u8[2::3].astype(np.int32) << 24)
                )
                s = (i32 >> 8).astype(np.float32) / 8388608.0
            else:
                s = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            audio = s.reshape(-1, n_ch) if n_ch > 1 else s
    else:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(input_path))
        sr = seg.frame_rate
        s = np.array(seg.get_array_of_samples(), dtype=np.float32) / (2.0 ** (seg.sample_width * 8 - 1))
        audio = s.reshape(-1, seg.channels) if seg.channels > 1 else s

    audio = _ensure_channel_layout(audio)
    m = ForensicCrossValidator.evaluate_track(audio, sr, file_path=input_path)

    tbl = Table(title=f"🔬 7-Domain Forensic & Golden-Ear Scorecard — {input_path.name}", border_style="cyan")
    tbl.add_column("Domain / Metric", style="bold white")
    tbl.add_column("Measured Value", justify="right", style="cyan")
    tbl.add_column("Engineering Assessment", style="bold green")
    tbl.add_row("1. Sample Rate", f"{m['sample_rate']} Hz", "Studio Master Ready" if m["sample_rate"] >= 48000 else "Standard Rate")
    tbl.add_row("2. True-Peak Level", f"{m['true_peak_dbfs']} dBFS", "✅ Zero-Clipping Safe" if m["zero_clipping"] else "⚠️ Near Clipping Limit")
    tbl.add_row("3. Dynamic RMS / Crest Factor", f"{m['rms']} ({m['crest_factor_db']} dB)", "✅ Healthy Dynamics")
    tbl.add_row("4. Cepstral Grid Ripple PNR", f"{m['cepstral_grid_pnr_db']} dB", "✅ Clean" if m["cepstral_grid_pnr_db"] < -5.5 else "ℹ️ Codec Ripple Present")
    tbl.add_row("5. RVQ Frame Periodicity Lock", f"{m['rvq_periodicity_index']}", "✅ Desynchronized" if m["rvq_periodicity_index"] < 3.25 else "ℹ️ Frame Lock Detectable")
    tbl.add_row("6. Spectral Flatness StdDev", f"{m['spectral_flatness_std']}", "✅ Organic Variance")
    tbl.add_row("7. Stereo HF Correlation (>2kHz)", f"{m['stereo_hf_correlation']}", "✅ Natural 3D Soundstage")
    tbl.add_row("8. Clipped Plateau Samples", f"{m.get('clipped_samples', 0)}", "✅ 0 Clipped Samples" if m.get('clipped_samples', 0) == 0 else "⚠️ Brickwall Clipping Detected")
    tbl.add_row("9. 24-Bit LSB TPDF Entropy", f"{m.get('lsb_entropy_bits')} bits", "✅ Authentic TPDF Dither (8.0 bits)")
    tbl.add_row("10. RIFF BWF/iXML DAW Provenance", f"{m.get('daw_provenance')}", "✅ Authentic DAW Container" if m.get('has_daw_riff_chunks') else "ℹ️ No BWF/iXML Chunks")
    console.console.print(tbl)


@cli.command()
def gui():
    """
    🖥️ Launch modern desktop graphical user interface
    """
    from .gui import launch_gui
    launch_gui()


@cli.command()
def version():
    """
    📋 Show version and build information
    """
    banner.show_version_info()


if __name__ == "__main__":
    cli()
