"""
Banner manager for WaZixwx-MMM Studio v7.0 CLI & Visual Telemetry.
"""

import random
import time
from rich.console import Console
from rich.panel import Panel
from .. import __version__


class BannerManager:
    """
    Manages studio banners and visual telemetry for WaZixwx-MMM Studio v7.0.
    """

    def __init__(self):
        self.console = Console()

        self.main_banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║  🎛️  WaZixwx-MMM STUDIO v{__version__:<8} (Melodic Metadata Massacrer)  ║
║      Ultra-Transparent Mastering & 7-Stem Separation Suite       ║
║  ──────────────────────────────────────────────────────────────  ║
║  🎼 Passband LSD < 0.06 dB • SNR > 40 dB • Phase-Exact 7-Stems   ║
╚══════════════════════════════════════════════════════════════════╝
"""

        self.skull_banner = self.main_banner
        self.matrix_banner = self.main_banner

        self.processing_banners = [
            """
🎛️ [bold cyan]Executing v7.0 Ultra-Transparent Acoustic Remodeling...[/bold cyan]
   [green]Passband LSD < 0.06 dB • >8kHz Phase Dispersion • Blumlein Stereo[/green]
            """,
            """
🔬 [bold magenta]Neutralizing 7-Domain AI Forensic Signatures...[/bold magenta]
   [cyan]Sub-Sample Phase Alignment • Spectral-Mirror BWE • Zero-Flutter Rubato[/cyan]
            """,
        ]

        self.success_banners = [
            """
✨ [bold green]STUDIO MASTERING & ZERO-TRACE EVASION COMPLETE![/bold green]
   ✓ Passband Log-Spectral Distance (LSD) < 0.06 dB (Zero Audible Coloration)
   ✓ True-Peak Ceiling <= -0.3 dBFS (Zero Clipping)
   ✓ 100% Mono-Compatible Blumlein 3D Stereo Field
            """,
        ]

    def show_main_banner(self):
        """Display the main studio banner."""
        panel = Panel.fit(self.main_banner.strip(), border_style="bold cyan", padding=(0, 1))
        self.console.print(panel)

    def show_processing_banner(self):
        """Show a processing status banner."""
        banner = random.choice(self.processing_banners)
        panel = Panel.fit(banner.strip(), border_style="cyan", padding=(0, 2))
        self.console.print(panel)

    def show_success_banner(self):
        """Show a completion banner."""
        banner = random.choice(self.success_banners)
        panel = Panel.fit(banner.strip(), border_style="green", padding=(0, 2))
        self.console.print(panel)

    def show_version_info(self):
        """Show version and architecture information."""
        version_info = f"""
[bold cyan]WaZixwx-MMM Studio v{__version__}[/bold cyan] (Melodic Metadata Massacrer)

📅 Architecture: v7.0 Ultra-Transparent Golden-Ear Mastering & 7-Stem Separation Engine
🔧 Python Runtime: 3.9+ | 64-Bit Float Internal DSP Precision
🎯 Target Engines: ArtifactNet, SONICS SpecTTTra, Deezer HOS, AudioSeal, SynthID
⚡ Core Modules:
   • >8kHz Zero-Coloration STFT Micro-Phase Disperser (0.000 dB Magnitude Invariance)
   • 8.0–16.5kHz Non-Peak Trough Micro-Smoothing (100% Fundamental Pitch Preservation)
   • Sub-Sample 0.15Hz Catmull-Rom Phase Alignment (< 0.003 cents pitch invariance)
   • Zero-Comb Complementary All-Pass Blumlein Stereo Field (L+R === 2M)
   • Spectral-Mirror Envelope-Gated Harmonic Bandwidth Extension (14.5kHz–22kHz)
        """
        panel = Panel.fit(version_info.strip(), border_style="cyan", padding=(1, 2))
        self.console.print(panel)

    def show_warning_banner(self, message: str):
        """Show warning banner."""
        panel = Panel.fit(f"[bold yellow]⚠️  {message}[/bold yellow]", border_style="yellow", padding=(0, 2))
        self.console.print(panel)

    def show_matrix_rain(self, duration: float = 0.5):
        """Simulated visual effect."""
        time.sleep(min(duration, 0.1))

    def show_dramatic_pause(self, message: str, duration: float = 0.3):
        """Display brief status message."""
        self.console.print(f"{message}...")
        time.sleep(min(duration, 0.1))

    def show_ascii_progress(self, current: int, total: int, width: int = 40):
        """Display ASCII progress bar."""
        percent = current / max(total, 1)
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        self.console.print(f"[{current}/{total}] [{bar}] {percent:.1%}")

    def show_audio_visualization(self, duration: float = 0.2):
        """Display brief audio bar visualization."""
        time.sleep(min(duration, 0.1))

    def show_easter_egg(self):
        """Show studio note."""
        panel = Panel.fit("🎼 WaZixwx-MMM Studio v7.0 — Crafted for Golden-Ear Musicians.", border_style="magenta")
        self.console.print(panel)

    def show_quote_banner(self, quote: str):
        """Show quote panel."""
        panel = Panel.fit(f"[italic cyan]💬 {quote}[/italic cyan]", border_style="cyan")
        self.console.print(panel)

    def show_error_banner(self, error: str):
        """Show error banner."""
        panel = Panel.fit(f"[bold red]❌ ERROR: {error}[/bold red]", border_style="red", padding=(0, 2))
        self.console.print(panel)
