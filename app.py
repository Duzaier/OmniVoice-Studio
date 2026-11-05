#!/usr/bin/env python3
"""
Cloud Entry Point (`app.py`) for Hugging Face Spaces, Render, Railway, and Docker deployments.
Automatically binds to `$PORT` (default 7860 on HF Spaces / 8001 locally) and serves both:
- `/`       -> OmniVoice Desktop & Mobile Unified Studio (Gradio)
- `/mobile` -> OmniVoice Standalone Mobile PWA
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
import gradio as gr
import uvicorn

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.ui.workstation_ui import build_workstation_ui
from omnivoice.ui.mobile_pwa import attach_mobile_pwa_routes


def create_cloud_app() -> FastAPI:
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    device = os.environ.get("OMNIVOICE_DEVICE", "cpu")

    engine = OmniVoiceEngine.get_instance(device=device)
    engine.initialize()

    demo = build_workstation_ui(engine)
    demo.queue(max_size=10)

    app = FastAPI(title="OmniVoice Studio & Mobile PWA")
    attach_mobile_pwa_routes(app, engine, port=port)
    app = gr.mount_gradio_app(app, demo, path="/")
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    app = create_cloud_app()
    uvicorn.run(app, host="0.0.0.0", port=port)
