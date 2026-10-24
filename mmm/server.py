"""
WaZixwx-MMM Web Studio v4.0.0 — Browser-Based Golden-Ear Mastering & 7-Domain Anti-AI Workstation.

Launch with: mmm server [--host 127.0.0.1] [--port 8778] [--max-size 500]
"""

import atexit
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from . import __version__

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"mp3", "wav", "flac"})
DEFAULT_MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500 MB
DEFAULT_STALE_AGE: int = 3600  # 1 hour

CSS_STYLES = """\
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:'Microsoft YaHei UI',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#090d16;color:#f1f5f9;min-height:100vh;
  display:flex;flex-direction:column;align-items:center;
}
header{text-align:center;padding:2.2rem 1rem 0.8rem}
header h1{font-size:1.75rem;color:#38bdf8;letter-spacing:0.5px}
header .subtitle{color:#94a3b8;font-size:0.9rem;margin-top:0.4rem}
main{width:100%;max-width:640px;padding:1rem}
.studio-badge{
  background:#111827;border:1px solid #1e293b;border-radius:8px;
  padding:0.75rem 1rem;font-size:0.8rem;color:#38bdf8;
  margin-bottom:1.2rem;line-height:1.5;text-align:center;
}
.dropzone{
  border:2px dashed #334155;border-radius:12px;padding:3rem 1.2rem;
  text-align:center;cursor:pointer;background:#111827;
  transition:border-color .2s,background .2s;
}
.dropzone:hover,.dropzone.dragover{border-color:#38bdf8;background:#1e293b}
.dropzone-icon{font-size:2.6rem;margin-bottom:0.6rem}
.dropzone p{color:#cbd5e1;font-size:0.95rem}
.dropzone .hint{color:#64748b;font-size:0.78rem;margin-top:0.35rem}
.dropzone .filename{color:#38bdf8;font-weight:600;margin-top:0.7rem;word-break:break-all}
.options-panel{
  background:#111827;border:1px solid #1e293b;border-radius:10px;
  padding:1.2rem;margin-top:1rem;
}
.option-group{display:flex;align-items:center;gap:0.8rem;margin-bottom:0.9rem}
.option-group label{color:#94a3b8;font-size:0.85rem;min-width:140px}
.option-group select{
  background:#090d16;color:#f1f5f9;border:1px solid #334155;
  border-radius:6px;padding:0.4rem 0.6rem;font-size:0.85rem;flex:1;
}
.option-group input[type=checkbox]{accent-color:#38bdf8}
.toggle-label{color:#cbd5e1;font-size:0.82rem}
.btn-primary{
  display:block;width:100%;padding:0.8rem;margin-top:0.6rem;
  background:#10b981;color:#fff;border:none;border-radius:8px;
  font-size:0.98rem;font-weight:600;cursor:pointer;transition:background .2s;
}
.btn-primary:hover{background:#059669}
.btn-primary:disabled{background:#1e293b;color:#475569;cursor:not-allowed}
.status-area{text-align:center;margin-top:1.5rem}
.spinner{
  width:38px;height:38px;border:3px solid #1e293b;border-top-color:#38bdf8;
  border-radius:50%;margin:0 auto 0.8rem;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
#statusText{color:#94a3b8;font-size:0.9rem}
.progress-bar{
  width:100%;height:6px;background:#1e293b;border-radius:3px;
  margin-top:0.8rem;overflow:hidden;
}
.progress-fill{
  height:100%;background:#38bdf8;border-radius:3px;
  transition:width .3s;width:0%;
}
.result-area{text-align:center;margin-top:1.5rem}
.success-icon{font-size:2.2rem;margin-bottom:0.5rem}
#resultText{color:#10b981;font-size:1rem;font-weight:600;margin-bottom:0.8rem}
.stats-panel{
  background:#111827;border:1px solid #1e293b;border-radius:8px;
  padding:1rem;margin-bottom:1.2rem;text-align:left;font-size:0.82rem;
}
.stats-panel .stat-row{
  display:flex;justify-content:space-between;padding:0.3rem 0;
  border-bottom:1px solid #1e293b;
}
.stats-panel .stat-row:last-child{border-bottom:none}
.stats-panel .stat-label{color:#94a3b8}
.stats-panel .stat-value{color:#38bdf8;font-weight:600}
.btn-download{
  display:inline-block;padding:0.7rem 1.8rem;background:#0284c7;color:#fff;
  border-radius:8px;text-decoration:none;font-weight:600;font-size:0.92rem;
  transition:background .2s;
}
.btn-download:hover{background:#0369a1}
.btn-secondary{
  display:inline-block;padding:0.55rem 1.3rem;margin-top:0.9rem;
  background:transparent;color:#94a3b8;border:1px solid #334155;
  border-radius:8px;font-size:0.85rem;cursor:pointer;transition:border-color .2s;
}
.btn-secondary:hover{border-color:#38bdf8;color:#38bdf8}
.error-area{text-align:center;margin-top:1.5rem}
.error-icon{font-size:2rem;margin-bottom:0.5rem}
#errorText{color:#f87171;font-size:0.9rem;margin-bottom:0.8rem}
footer{margin-top:auto;padding:1.5rem;text-align:center;color:#475569;font-size:0.75rem}
"""

JS_APP = """\
(function(){
  const $ = id => document.getElementById(id);
  const dropzone = $('dropzone');
  const fileInput = $('fileInput');
  const options = $('options');
  const status = $('status');
  const result = $('result');
  const error = $('error');
  let selectedFile = null;

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp3','wav','flac'].includes(ext)) {
      showError('Unsupported file type. Please upload MP3, WAV, or FLAC.');
      return;
    }
    const maxBytes = parseInt(document.body.dataset.maxSize || '524288000', 10);
    if (file.size > maxBytes) {
      showError('File exceeds maximum size: ' + formatBytes(maxBytes));
      return;
    }
    selectedFile = file;
    let fn = dropzone.querySelector('.filename');
    if (!fn) { fn = document.createElement('p'); fn.className='filename'; dropzone.querySelector('.dropzone-content').appendChild(fn); }
    fn.textContent = file.name + ' (' + formatBytes(file.size) + ')';
    options.hidden = false;
    result.hidden = true;
    error.hidden = true;
  }

  $('processBtn').addEventListener('click', processFile);

  function processFile() {
    if (!selectedFile) return;
    const fmt = $('formatSelect').value;
    const paranoid = $('paranoidToggle').checked;

    options.hidden = true;
    status.hidden = false;
    result.hidden = true;
    error.hidden = true;
    $('statusText').textContent = 'Uploading audio stream...';
    $('progressFill').style.width = '0%';
    $('processBtn').disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('format', fmt);
    formData.append('paranoid', paranoid ? 'true' : 'false');

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 50);
        $('progressFill').style.width = pct + '%';
        $('statusText').textContent = 'Uploading... ' + pct*2 + '%';
      }
    });
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        if (data.success) {
          showResult(data);
        } else {
          showError(data.error || 'Processing failed.');
        }
      } else if (xhr.status === 429) {
        showError('Studio engine is currently processing another track. Please wait.');
      } else {
        let msg = 'Studio server error.';
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e) {}
        showError(msg);
      }
      $('processBtn').disabled = false;
    });
    xhr.addEventListener('error', () => {
      showError('Network connection lost. Please verify the local server is running.');
      $('processBtn').disabled = false;
    });

    xhr.open('POST', '/api/upload');
    xhr.send(formData);

    xhr.upload.addEventListener('loadend', () => {
      $('statusText').textContent = 'Executing Golden-Ear Acoustic Remodeling...';
      $('progressFill').style.width = '55%';
      let p = 55;
      const iv = setInterval(() => {
        if (p < 95) { p += Math.random() * 4; $('progressFill').style.width = Math.min(p, 95) + '%'; }
      }, 450);
      xhr.addEventListener('loadend', () => { clearInterval(iv); $('progressFill').style.width = '100%'; });
    });
  }

  function showResult(data) {
    status.hidden = true;
    result.hidden = false;
    $('resultText').textContent = 'Studio Remodeling & Zero-Trace Evasion Complete!';
    const stats = data.stats || {};
    let html =
      '<div class="stat-row"><span class="stat-label">Metadata Tags Stripped</span><span class="stat-value">' + (stats.metadata_removed || 0) + '</span></div>' +
      '<div class="stat-row"><span class="stat-label">Processing Time</span><span class="stat-value">' + formatTime(stats.processing_time) + '</span></div>';
    if (stats.true_peak_dbfs !== undefined) {
      html += '<div class="stat-row"><span class="stat-label">True-Peak Ceiling</span><span class="stat-value">' + stats.true_peak_dbfs + ' dBFS (Zero-Clip)</span></div>';
    }
    if (stats.rms !== undefined) {
      html += '<div class="stat-row"><span class="stat-label">Dynamic RMS / Crest</span><span class="stat-value">' + stats.rms + ' (' + (stats.crest_factor_db || '') + ' dB)</span></div>';
    }
    $('statsPanel').innerHTML = html;
    $('downloadLink').href = '/api/download/' + data.download_token;
    $('downloadLink').download = data.filename || 'studio_master.wav';
  }

  function showError(msg) {
    status.hidden = true;
    options.hidden = true;
    error.hidden = false;
    $('errorText').textContent = msg;
    $('processBtn').disabled = false;
  }

  $('resetBtn').addEventListener('click', resetUI);
  $('retryBtn').addEventListener('click', resetUI);

  function resetUI() {
    selectedFile = null;
    fileInput.value = '';
    const fn = dropzone.querySelector('.filename');
    if (fn) fn.remove();
    options.hidden = true;
    status.hidden = true;
    result.hidden = true;
    error.hidden = true;
    $('processBtn').disabled = false;
  }

  function formatBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
  }
  function formatTime(s) {
    if (s == null) return 'N/A';
    return parseFloat(s).toFixed(1) + 's';
  }
})();
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WaZixwx-MMM Web Studio v4.0.0</title>
<style>{css}</style>
</head>
<body data-max-size="{max_size}">
<header>
  <h1>🎛️ WaZixwx-MMM Web Studio v4.0</h1>
  <p class="subtitle">Golden-Ear Master Restoration &amp; 7-Domain Anti-AI Acoustic Workstation</p>
</header>
<main>
  <div class="studio-badge">
    🎼 64-Bit Float DSP &bull; 0.000 dB All-Pass Phase Dispersion &bull; Zero-Comb Blumlein Stereo &bull; -0.3 dBFS True-Peak Ceiling
  </div>

  <div id="dropzone" class="dropzone">
    <div class="dropzone-content">
      <div class="dropzone-icon">&#127925;</div>
      <p>拖拽音频文件至此，或点击选择音轨 (Drag &amp; Drop Audio Track)</p>
      <p class="hint">支持 MP3, WAV, FLAC &middot; 最大文件体积 {max_size_mb} MB</p>
      <input type="file" id="fileInput" accept=".mp3,.wav,.flac" hidden>
    </div>
  </div>

  <div id="options" class="options-panel" hidden>
    <div class="option-group">
      <label for="formatSelect">输出格式 (Format):</label>
      <select id="formatSelect">
        <option value="preserve" selected>保持原格式 (Preserve Format)</option>
        <option value="wav">24-Bit / 48kHz Studio WAV</option>
        <option value="mp3">High-Bitrate MP3</option>
      </select>
    </div>
    <div class="option-group">
      <label for="paranoidToggle">深度重构 (Deep Mode):</label>
      <input type="checkbox" id="paranoidToggle" checked>
      <span class="toggle-label">启用 11 阶全维声学重构与零相位倒谱净化</span>
    </div>
    <button id="processBtn" class="btn-primary">⚡ 开始母带级重构 (Start Studio Processing)</button>
  </div>

  <div id="status" class="status-area" hidden>
    <div class="spinner" id="spinner"></div>
    <p id="statusText">正在执行声学重构...</p>
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
  </div>

  <div id="result" class="result-area" hidden>
    <div class="success-icon">&#9989;</div>
    <p id="resultText"></p>
    <div id="statsPanel" class="stats-panel"></div>
    <a id="downloadLink" class="btn-download" href="#">📥 下载重构后的母带音轨 (Download Master)</a><br>
    <button id="resetBtn" class="btn-secondary">继续处理下一首 (Process Another Track)</button>
  </div>

  <div id="error" class="error-area" hidden>
    <div class="error-icon">&#10060;</div>
    <p id="errorText"></p>
    <button id="retryBtn" class="btn-secondary">重试 (Try Again)</button>
  </div>
</main>
<footer>WaZixwx-MMM Studio v4.0.0 &mdash; Golden-Ear Acoustic Remodeling &amp; Anti-AI Forensics Suite</footer>
<script>{js}</script>
</body>
</html>
"""


def _validate_extension(filename: str) -> bool:
    """Check file extension against the allowlist."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def _cleanup_old_files(directory: Path, max_age: int = DEFAULT_STALE_AGE) -> None:
    """Remove files older than *max_age* seconds from *directory*."""
    now = time.time()
    try:
        for entry in directory.iterdir():
            if entry.is_file():
                age = now - entry.stat().st_mtime
                if age > max_age:
                    entry.unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_download_registry(
    registry: Dict[str, Dict[str, Any]], max_age: int = DEFAULT_STALE_AGE
) -> None:
    """Drop stale download tokens and unlink stale files."""
    now = time.time()
    stale_tokens = [
        token
        for token, entry in registry.items()
        if now - float(entry.get("created", 0)) > max_age
    ]
    for token in stale_tokens:
        entry = registry.pop(token, None)
        if entry:
            Path(str(entry.get("path", ""))).unlink(missing_ok=True)


def create_app(
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Flask:
    """Create and configure the WaZixwx-MMM Web Studio Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_file_size
    app.config["SECRET_KEY"] = os.urandom(32)

    temp_dir = Path(tempfile.mkdtemp(prefix="mmm_server_"))
    app.config["UPLOAD_FOLDER"] = temp_dir
    atexit.register(shutil.rmtree, str(temp_dir), True)

    app.config["PROCESSING_LOCK"] = threading.Lock()
    app.config["DOWNLOAD_REGISTRY"]: Dict[str, Dict[str, Any]] = {}

    max_size_mb = max_file_size // (1024 * 1024)

    @app.route("/")
    def index() -> Response:
        html = HTML_TEMPLATE.format(
            css=CSS_STYLES,
            js=JS_APP,
            max_size=max_file_size,
            max_size_mb=max_size_mb,
        )
        return Response(html, mimetype="text/html")

    @app.route("/api/status")
    def api_status() -> tuple:
        _cleanup_download_registry(app.config["DOWNLOAD_REGISTRY"])
        lock: threading.Lock = app.config["PROCESSING_LOCK"]
        busy = not lock.acquire(blocking=False)
        if not busy:
            lock.release()
        return jsonify({
            "busy": busy,
            "version": __version__,
            "max_file_size_mb": max_size_mb,
        }), 200

    @app.route("/api/upload", methods=["POST"])
    def api_upload() -> tuple:
        lock: threading.Lock = app.config["PROCESSING_LOCK"]
        _cleanup_download_registry(app.config["DOWNLOAD_REGISTRY"])

        if not lock.acquire(blocking=False):
            return jsonify({"error": "Server is busy processing another file. Please wait."}), 429

        try:
            return _handle_upload(app)
        except Exception:
            app.logger.exception("Upload handler failed")
            return jsonify({"error": "Sanitization failed. Please try a different file."}), 500
        finally:
            lock.release()

    @app.route("/api/download/<token>")
    def api_download(token: str) -> tuple:
        registry: dict = app.config["DOWNLOAD_REGISTRY"]
        _cleanup_download_registry(registry)
        entry = registry.pop(token, None)

        if entry is None:
            return jsonify({"error": "File not found or expired."}), 404

        file_path = Path(entry["path"])
        if not file_path.exists():
            return jsonify({"error": "File not found or expired."}), 404

        response = send_file(
            str(file_path),
            as_attachment=True,
            download_name=entry["filename"],
        )
        response.call_on_close(lambda: file_path.unlink(missing_ok=True))
        return response

    @app.errorhandler(413)
    def _too_large(e):
        _ = e
        return jsonify({"error": f"File too large. Maximum size: {max_size_mb} MB."}), 413

    return app


def _handle_upload(app: Flask) -> tuple:
    """Process an uploaded file inside the processing lock."""
    temp_dir: Path = app.config["UPLOAD_FOLDER"]
    registry: dict = app.config["DOWNLOAD_REGISTRY"]

    _cleanup_old_files(temp_dir)
    _cleanup_download_registry(registry)

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400

    safe_name = secure_filename(f.filename)
    if not safe_name or not _validate_extension(safe_name):
        return jsonify({"error": "Unsupported file type. Use MP3, WAV, or FLAC."}), 400

    output_format = request.form.get("format", "preserve")
    if output_format not in ("preserve", "mp3", "wav"):
        output_format = "preserve"

    paranoid = request.form.get("paranoid", "false").lower() == "true"

    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    input_path = temp_dir / unique_name
    f.save(str(input_path))

    try:
        from .preserving_sanitizer import preserving_sanitize

        fmt = None if output_format == "preserve" else output_format
        result = preserving_sanitize(
            input_file=input_path,
            output_file=None,
            paranoid_mode=paranoid,
            threat_count=0,
            output_format=fmt,
        )

        if not result.get("success"):
            return jsonify({"error": "Sanitization failed."}), 500

        output_path = Path(result["output_file"])
        token = uuid.uuid4().hex
        stem = Path(safe_name).stem
        ext = output_path.suffix
        download_name = f"{stem}_clean{ext}"

        registry[token] = {
            "path": str(output_path),
            "filename": download_name,
            "created": time.time(),
        }

        return jsonify({
            "success": True,
            "download_token": token,
            "filename": download_name,
            "stats": result.get("stats", {}),
        }), 200

    finally:
        input_path.unlink(missing_ok=True)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8778,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> None:
    """Create and run the WaZixwx-MMM Web Studio server."""
    from .ui.console import ConsoleManager
    from .ui.banners import BannerManager

    console = ConsoleManager()
    banner = BannerManager()

    banner.show_main_banner()

    if host != "127.0.0.1":
        console.warning(
            f"Binding to {host} — Web Studio will be accessible from the local network."
        )

    max_mb = max_file_size // (1024 * 1024)
    console.success(f"Starting WaZixwx-MMM Web Studio v{__version__} at http://{host}:{port}")
    console.info(f"   Max upload size: {max_mb} MB | Press Ctrl+C to stop\n")

    app = create_app(max_file_size=max_file_size)
    app.run(host=host, port=port, debug=False, threaded=True)
