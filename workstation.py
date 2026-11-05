#!/usr/bin/env python3
"""
OmniVoice Production Workstation & Mobile PWA Launcher (V2.26 Multi-Environment Cloud Edition).
Starts the FastAPI + Gradio server in production mode with:
- `/`            -> Swiss-Minimalist Unified Studio (Desktop + Mobile responsive)
- `/mobile`      -> Standalone Installable Mobile PWA (Service Worker + Offline Shell + Full Feature Parity)
- Public HTTPS   -> Automatic dual HTTPS tunnels (`localhost.run` + `pinggy.io`) for instant global access
"""

import os
import re
import sys
import site
import time
import argparse
import logging
import subprocess
import threading
import webbrowser
from pathlib import Path
from typing import Optional

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent if _THIS_FILE.parent.name == "cli" else _THIS_FILE.parent
if not (_PROJECT_ROOT / "omnivoice").exists():
    _PROJECT_ROOT = Path(r"D:\ALL PROJECT\OJT PROJECT\OmniVoice")

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_VENV_SITE = _PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if _VENV_SITE.exists():
    site.addsitedir(str(_VENV_SITE))

# Protect C: drive before importing anything heavy
os.environ["TEMP"] = str(_PROJECT_ROOT / "cache" / "temp")
os.environ["TMP"] = str(_PROJECT_ROOT / "cache" / "temp")
os.environ["TMPDIR"] = str(_PROJECT_ROOT / "cache" / "temp")
os.environ["GRADIO_TEMP_DIR"] = str(_PROJECT_ROOT / "cache" / "gradio")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI
import gradio as gr
import uvicorn

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.ui.workstation_ui import build_workstation_ui
from omnivoice.ui.mobile_pwa import attach_mobile_pwa_routes, get_lan_ips, set_public_tunnel_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OmniVoiceWorkstation")


def _ensure_ssh_key(project_root: Path) -> Optional[Path]:
    key_path = project_root / "cache" / "tunnel_key"
    if key_path.exists():
        return key_path
    keygen = r"C:\WINDOWS\System32\OpenSSH\ssh-keygen.exe"
    if not Path(keygen).exists():
        return None
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([keygen, "-t", "ed25519", "-f", str(key_path), "-N", ""], capture_output=True, timeout=10)
        if key_path.exists():
            return key_path
    except Exception:
        pass
    return None


def _run_single_ssh_tunnel(cmd: list, pattern: re.Pattern, label: str, url_file: Path) -> None:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout:
            for line in proc.stdout:
                m = pattern.search(line)
                if m:
                    pub_url = m.group(1).strip().rstrip("/")
                    set_public_tunnel_url(pub_url)
                    try:
                        existing = url_file.read_text(encoding="utf-8") if url_file.exists() else ""
                        if pub_url not in existing:
                            line_out = f"[{label}] Desktop: {pub_url} | Mobile: {pub_url}/mobile\n"
                            with open(url_file, "a", encoding="utf-8") as f:
                                f.write(line_out)
                            scratch_url_file = Path(r"C:\Users\Duzaier\.gemini\antigravity\brain\933a6da8-dff8-47f0-b259-95c8ca63bf7a\scratch\public_urls.txt")
                            if scratch_url_file.parent.exists():
                                with open(scratch_url_file, "a", encoding="utf-8") as sf:
                                    sf.write(line_out)
                    except Exception:
                        pass
                    print("\n" + "=" * 72, flush=True)
                    print(f"  🌐 [{label}] PUBLIC DESKTOP STUDIO URL : {pub_url}", flush=True)
                    print(f"  📱 [{label}] PUBLIC MOBILE APP URL     : {pub_url}/mobile", flush=True)
                    print("=" * 72 + "\n", flush=True)
    except Exception as e:
        logger.debug(f"Tunnel {label} skipped: {e}")


def _start_https_tunnels(port: int, project_root: Path) -> None:
    """Starts dual zero-config SSH reverse tunnels (localhost.run + pinggy.io) with an explicit SSH key."""
    ssh_bin = r"C:\WINDOWS\System32\OpenSSH\ssh.exe"
    if not Path(ssh_bin).exists():
        return

    key_path = _ensure_ssh_key(project_root)
    key_args = ["-i", str(key_path)] if key_path else []

    url_file = project_root / "logs" / "public_urls.txt"
    url_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        url_file.write_text("", encoding="utf-8")
    except Exception:
        pass

    # Tunnel 1: localhost.run
    cmd_lhr = [
        ssh_bin,
        *key_args,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        "-o", "ServerAliveInterval=30",
        "-R", f"80:127.0.0.1:{port}",
        "nokey@localhost.run",
    ]
    threading.Thread(
        target=_run_single_ssh_tunnel,
        args=(cmd_lhr, re.compile(r"(https://[a-zA-Z0-9.-]+\.lhr\.life)"), "localhost.run", url_file),
        daemon=True,
    ).start()

    # Tunnel 2: pinggy.io
    cmd_pinggy = [
        ssh_bin,
        *key_args,
        "-p", "443",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        "-o", "ServerAliveInterval=30",
        f"-R0:127.0.0.1:{port}",
        "a.pinggy.io",
    ]
    threading.Thread(
        target=_run_single_ssh_tunnel,
        args=(cmd_pinggy, re.compile(r"(https://[a-zA-Z0-9.-]+\.pinggy\.link)"), "pinggy.io", url_file),
        daemon=True,
    ).start()


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Production Workstation & Mobile App")
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001)")
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0)")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, xpu, mps)")
    parser.add_argument("--share", action="store_true", help="Create Gradio shareable public link")
    parser.add_argument("--no-tunnel", action="store_true", help="Disable automatic HTTPS public tunnels")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    lan_ips = get_lan_ips()
    primary_lan = lan_ips[0] if lan_ips else "localhost"

    print("\n" + "=" * 72, flush=True)
    print("  🎙️🎬📱 OMNIVOICE PRODUCTION STUDIO & MOBILE PWA (V2.26)", flush=True)
    print("  Optimized for Dell Inspiron 15 3515 (AMD Ryzen 5 3450U + Vega 8)", flush=True)
    print("=" * 72, flush=True)

    # 1. Initialize Engine
    engine = OmniVoiceEngine.get_instance(device=args.device)
    engine.initialize()

    # 2. Build Gradio UI
    demo = build_workstation_ui(engine)
    demo.queue(max_size=10)

    print("\n" + "-" * 72, flush=True)
    print(f"  💻 Desktop Studio (Local)  : http://localhost:{args.port}", flush=True)
    print(f"  📱 Mobile PWA (Local)      : http://localhost:{args.port}/mobile", flush=True)
    print(f"  📶 Desktop Studio (Wi-Fi)  : http://{primary_lan}:{args.port}", flush=True)
    print(f"  📲 Mobile PWA (Wi-Fi)      : http://{primary_lan}:{args.port}/mobile", flush=True)
    print("-" * 72 + "\n", flush=True)

    # 3. Start background HTTPS public tunnels for global Desktop & Mobile access
    if not args.no_tunnel:
        _start_https_tunnels(args.port, _PROJECT_ROOT)

    # 4. Create FastAPI App with `/mobile` PWA, `/sw.js`, `/manifest.json`, and `/` Gradio Studio
    app = FastAPI(title="OmniVoice Production Studio & Mobile PWA")
    attach_mobile_pwa_routes(app, engine, port=args.port)
    app = gr.mount_gradio_app(app, demo, path="/")

    if not args.no_browser:
        def _open_browser():
            time.sleep(1.8)
            try:
                webbrowser.open(f"http://localhost:{args.port}")
            except Exception:
                pass
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=args.ip, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
