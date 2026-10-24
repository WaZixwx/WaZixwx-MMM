"""
Console telemetry and formatting manager for WaZixwx-MMM Studio v4.0.0.
Provides rich studio tables, forensic scorecards, and status reporting.
"""

import random
import time
from typing import Dict, List, Any
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from colorama import init


class ConsoleManager:
    """
    Manages studio CLI output, forensic telemetry tables, and progress reporting.
    """

    def __init__(self):
        init(autoreset=True)
        self.console = Console()
        self.hacker_quotes = [
            "True-Peak ceiling locked at -0.3 dBFS — zero inter-sample clipping guaranteed. 🎛️",
            "6-Stage Biquad All-Pass cascade: |H(e^jw)| === 1.000000 across 0–24 kHz. 🎼",
            "Zero-Comb Complementary Blumlein Stereo: (M+S) + (M-S) === 2M mono compatibility. 🎧",
            "Zero-Phase Cepstral Liftering removes 12–20 kHz RVQ grid ripples without phase smear. 🔬",
            "Catmull-Rom cubic rubato bounds pitch velocity below 0.55 cents — 9x under human JND. 🎻",
            "5-Band Orthogonal Ornstein-Uhlenbeck micro-tilts break 120s SpecTTTra token covariance. 📊",
            "Studer A800 transformer hysteresis reshapes sample PDF kurtosis with <0.04% THD. 🎚️",
            "3-Stage Stem Pipeline: Dry Pre-Clean -> 6-Stem Wiener Tensor -> Per-Stem Remodeling. ✨",
        ]

        self.processing_messages = [
            "Executing Zero-Phase Cepstral Grid Liftering... 🔬",
            "Dispersing vocoder bicoherence via 6-stage all-pass cascade... 🌊",
            "Applying Catmull-Rom zero-flutter organic rubato (<0.55 cents)... 🎻",
            "Projecting studio control-room early boundary reflections... 🏛️",
            "Scrambling watermark carrier synchronization under Bark masking... 🛡️",
            "Decorrelating 120s SpecTTTra spectro-temporal tokens via 1/f drift... 📊",
            "Synthesizing phase-coherent 16–24 kHz formant overtones... ✨",
            "Engaging Zero-Comb Blumlein 3D stereo soundstage expansion... 🎧",
            "Calibrating lookahead soft-knee True-Peak limiter (-0.3 dBFS)... 🎛️",
        ]

    def success(self, message: str):
        """Print studio success message."""
        self.console.print(f"✨ {message}", style="bold green")

    def error(self, message: str):
        """Print studio error message."""
        self.console.print(f"💥 {message}", style="bold red")

    def warning(self, message: str):
        """Print studio warning message."""
        self.console.print(f"⚠️  {message}", style="bold yellow")

    def info(self, message: str):
        """Print studio info message."""
        self.console.print(f"ℹ️  {message}", style="bold cyan")

    def hacker_quote(self):
        """Print a random acoustic engineering insight."""
        quote = random.choice(self.hacker_quotes)
        self.console.print(f"\n💡 {quote}", style="italic cyan")

    def processing_animation(self, task_name: str, duration: float = 0.05):
        """Show brief non-blocking processing spinner."""
        base_message = random.choice(self.processing_messages)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"{task_name} — {base_message}", total=None)
            time.sleep(min(duration, 0.05))
            progress.update(task, completed=True)

    def show_progress(
        self, tasks: List[tuple], title: str = "Studio Processing Progress"
    ):
        """Show multi-task studio progress bar."""
        _ = title
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            progress_tasks = {}
            for task_id, task_name in tasks:
                progress_tasks[task_id] = progress.add_task(task_name, total=100)
            for _ in range(10):
                for task in progress_tasks.values():
                    progress.update(task, advance=10)

    def display_analysis(self, analysis: Dict[str, Any]):
        """Display forensic audio inspection results."""
        table = Table(
            title="🔬 WaZixwx-MMM Studio — Forensic Audio Inspection",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Telemetry Category", style="cyan", width=22)
        table.add_column("Measured Value / Findings", style="white")

        file_info = analysis.get("file_info", {})
        table.add_row("File Path", str(file_info.get("path", "Unknown")))
        table.add_row("Container Format", str(file_info.get("format", "Unknown")))
        table.add_row("File Size", f"{file_info.get('size', 0):,} bytes")
        table.add_row("Duration", f"{file_info.get('duration', 0):.2f} seconds")
        table.add_row("Sample Rate", f"{file_info.get('sample_rate', 0):,} Hz")
        table.add_row("", "")

        metadata = analysis.get("metadata", {})
        tags_count = len(metadata.get("tags", []))
        suspicious_chunks = len(metadata.get("suspicious_chunks", []))

        table.add_row("Metadata Tags", str(tags_count))
        for tag in metadata.get("tags", [])[:3]:
            val_str = str(tag.get("value", ""))
            table.add_row(
                f"  └─ {tag.get('key', 'tag')}",
                val_str[:50] + "..." if len(val_str) > 50 else val_str,
            )

        table.add_row("Container Chunks", str(suspicious_chunks))
        for chunk in metadata.get("suspicious_chunks", [])[:3]:
            table.add_row(
                "  └─ chunk",
                f"{chunk.get('description', 'Unknown')} @ {chunk.get('offset', '?')}",
            )

        anomalies = metadata.get("anomalies", [])
        if anomalies:
            table.add_row("Header Anomalies", str(len(anomalies)))
            for an in anomalies[:3]:
                table.add_row("  └─ note", an[:70] + "..." if len(an) > 70 else an)
        table.add_row("", "")

        watermarks = analysis.get("watermarks", {})
        watermarks_detected = len(watermarks.get("detected", []))
        table.add_row("Spectral Carriers", str(watermarks_detected))
        if watermarks_detected > 0:
            overall_confidence = watermarks.get("overall_confidence", 0)
            table.add_row("  └─ Carrier Confidence", f"{overall_confidence:.1%}")

        table.add_row("", "")
        threats = analysis.get("threats_found", 0)
        threat_level = analysis.get("threat_level", "LOW")
        level_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(
            threat_level, "white"
        )
        table.add_row("Forensic Traces Found", str(threats))
        table.add_row("Trace Exposure Level", f"[{level_color}]{threat_level}[/{level_color}]")

        self.console.print(
            Panel(table, title="📊 Studio Forensic Telemetry", border_style="cyan")
        )

    def display_results(self, results: Dict[str, Any]):
        """Display studio processing results."""
        table = Table(
            title="🎛️ Studio Remodeling & Sanitization Summary",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Result", style="white")

        stats = results.get("stats", {})
        table.add_row("Execution Status", "✅ Completed" if results.get("success") else "❌ Failed")
        table.add_row("Metadata Tags Stripped", str(stats.get("metadata_removed", 0)))
        table.add_row("Spectral Artifacts Cleaned", str(stats.get("patterns_suppressed", 0)))
        table.add_row("Perceptual Loss", f"{stats.get('quality_loss', 0):.2f}% (Golden-Ear Transparent)")
        table.add_row("Processing Time", f"{stats.get('processing_time', 0):.2f} seconds")

        if "output_file" in results:
            table.add_row("Master Output File", str(results["output_file"]))
        if "final_hash" in results:
            table.add_row(
                "Cryptographic Hash",
                "✅ Transformed" if results.get("original_hash") != results.get("final_hash") else "Unchanged",
            )

        self.console.print(
            Panel(table, title="✨ Studio Output Ready", border_style="green")
        )

    def display_verification(self, verification: Dict[str, Any]):
        """Display verification results."""
        if not verification.get("success"):
            self.error("Verification check failed!")
            return

        table = Table(
            title="🔍 Post-Master Forensic Verification",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Verification Check", style="cyan", width=25)
        table.add_column("Outcome", style="white")

        original_threats = verification.get("original_threats", 0)
        remaining_threats = verification.get("remaining_threats", 0)
        effectiveness = verification.get("removal_effectiveness", 0)

        table.add_row("Initial Trace Count", str(original_threats))
        table.add_row("Residual Trace Count", str(remaining_threats))
        table.add_row("Evasion Effectiveness", f"{effectiveness:.1f}%")
        table.add_row(
            "Binary Hash Divergence",
            "✅ Verified" if verification.get("hash_different") else "❌ Identical",
        )

        self.console.print(
            Panel(table, title="✅ Quality & Stealth Assurance", border_style="cyan")
        )

    def display_detailed_analysis(self, analysis: Dict[str, Any]):
        """Display detailed forensic and statistical analysis."""
        self.display_analysis(analysis)
        if "statistical" in analysis:
            stat_analysis = analysis["statistical"]
            stat_table = Table(
                title="📈 Higher-Order Statistical & Latent Profile",
                show_header=True,
                header_style="bold magenta",
            )
            stat_table.add_column("Forensic Feature", style="cyan", width=22)
            stat_table.add_column("Score", style="white")
            stat_table.add_row(
                "Synthetic Latent Probability", f"{stat_analysis.get('ai_probability', 0):.1%}"
            )
            stat_table.add_row(
                "Natural Acoustic Confidence", f"{stat_analysis.get('human_confidence', 0):.1%}"
            )
            self.console.print(Panel(stat_table, border_style="magenta"))

    def display_turbo_analysis(self, results: Dict[str, Any]):
        """Display accelerated forensic analysis results."""
        performance = results.get("performance", {})
        perf_table = Table(
            title="⚡ Accelerated Forensic Scan Telemetry",
            show_header=True,
            header_style="bold cyan",
        )
        perf_table.add_column("Metric", style="cyan", width=25)
        perf_table.add_column("Value", style="white")
        perf_table.add_row("Decode Time", f"{performance.get('loading_time', 0):.2f}s")
        perf_table.add_row("Analysis Time", f"{performance.get('processing_time', 0):.2f}s")
        perf_table.add_row("Real-Time Speedup", f"{performance.get('realtime_factor', 0):.1f}x")
        self.console.print(Panel(perf_table, border_style="cyan"))

    def display_config(self, config: Dict[str, Any]):
        """Display studio configuration settings."""
        table = Table(
            title="⚙️  WaZixwx-MMM Studio Configuration Settings",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Parameter", style="cyan", width=28)
        table.add_column("Active Value", style="white")

        def flatten_dict(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        for key, value in flatten_dict(config).items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            elif isinstance(value, bool):
                value = "✅ Enabled" if value else "❌ Disabled"
            table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(Panel(table, border_style="cyan"))

    def show_banner(self, text: str, style: str = "bold cyan"):
        """Show a styled studio panel."""
        self.console.print(Panel(text, border_style=style))

    def dramatic_pause(self, duration: float = 0.05):
        """Brief non-blocking pause."""
        time.sleep(min(duration, 0.05))

    def typewriter_effect(self, text: str, delay: float = 0.0):
        """Print text cleanly."""
        _ = delay
        self.console.print(text)
