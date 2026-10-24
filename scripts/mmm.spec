# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
project_root = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()

hidden_imports = [
    "mmm",
    "mmm.cli",
    "mmm.gui",
    "mmm.server",
    "mmm.acoustic_remodeling",
    "mmm.master_restoration",
    "mmm.pipeline",
    "mmm.super_resolution",
    "mmm.preserving_sanitizer",
    "mmm.effective_sanitizer",
    "mmm.fast_sanitizer",
    "mmm.optimized_processor",
    "mmm.turbo_analysis",
    "mmm.config",
    "mmm.config.config_manager",
    "mmm.config.defaults",
    "mmm.core",
    "mmm.core.audio_sanitizer",
    "mmm.core.file_processor",
    "mmm.detection",
    "mmm.detection.metadata_scanner",
    "mmm.detection.statistical_analyzer",
    "mmm.detection.watermark_detector",
    "mmm.sanitization",
    "mmm.sanitization.metadata_cleaner",
    "mmm.sanitization.spectral_cleaner",
    "mmm.sanitization.fingerprint_remover",
    "mmm.sanitization.psychoacoustic_sanitizer",
    "mmm.ui",
    "mmm.ui.banners",
    "mmm.ui.console",
    "mmm.ui.progress",
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.ndimage",
    "click",
    "rich",
    "colorama",
    "yaml",
    "pydub",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
]

a = Analysis(
    [str(project_root / "mmm" / "gui.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tests"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WaZixwx-MMM-Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="WaZixwx-MMM-Studio",
)
