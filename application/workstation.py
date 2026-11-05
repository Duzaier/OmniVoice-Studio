#!/usr/bin/env python3
"""
OmniVoice Production Workstation Launcher.
Starts the server with persistent model in memory and opens browser UI.
"""

import os
import sys
import argparse
import logging
import webbrowser

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure HF_ENDPOINT is set for reliable downloads
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.ui.workstation_ui import build_workstation_ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OmniVoiceWorkstation")


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Production Workstation")
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001)")
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0)")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, xpu, mps)")
    parser.add_argument("--share", action="store_true", help="Create Gradio shareable public link")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("      🎙️ OMNIVOICE PRODUCTION WORKSTATION STARTUP")
    print("      Theme: Dark Mode (Quality V5 Edition)")
    print("=" * 60)
    
    # 1. Initialize Engine (Load pure model into RAM once)
    engine = OmniVoiceEngine.get_instance(device=args.device)
    engine.initialize()

    # 2. Build UI
    demo = build_workstation_ui(engine)

    # 3. Launch Server
    logger.info(f"Starting Workstation Web UI at http://localhost:{args.port}")
    demo.queue(max_size=10).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
