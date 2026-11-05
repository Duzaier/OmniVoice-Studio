#!/usr/bin/env python3
"""
OmniVoice Production Mobile App (PWA) & REST API (V2.25).
Complete standalone mobile application at `/mobile` with:
- Offline-ready Service Worker (`/sw.js`) & dynamically generated PNG App Icons (`/icon-192.png`, `/icon-512.png`)
- Swiss Minimalist single-screen interface (Zero duplication, progressive disclosure via <details> drawers)
- 100% Feature Parity with Desktop Studio:
  * TTS + Video MP4 generation (all 5 FFmpeg presets, 720p/1080p, 24/30fps)
  * Custom background image upload & existing audio-to-video muxing from phone gallery
  * One-shot voice cloning & Persistent Voice Library (Create / Delete)
  * In-browser WebAudio WAV Microphone Recorder + File Capture fallback
  * Native iOS/Android Web Share API (`navigator.share` for Zalo, TikTok, Messenger, Files)
  * Live Public HTTPS Tunnel & Wi-Fi LAN connection modal
"""

import io
import os
import time
import socket
import shutil
from pathlib import Path
from typing import Optional, List
import soundfile as sf
from PIL import Image, ImageDraw

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.core.hardware import get_detailed_resource_usage
from omnivoice.core.text_preprocessor import TextPreprocessor

# Global store for active public HTTPS tunnel URL (populated by launcher if tunnel is active)
PUBLIC_TUNNEL_URL: Optional[str] = None


def set_public_tunnel_url(url: Optional[str]) -> None:
    global PUBLIC_TUNNEL_URL
    PUBLIC_TUNNEL_URL = url


def get_lan_ips() -> List[str]:
    ips = []
    try:
        _, _, addr_list = socket.gethostbyname_ex(socket.gethostname())
        for ip in addr_list:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.append(ip)
    except Exception:
        pass
    return ips or ["localhost"]


def _generate_app_icon_png(size: int = 512) -> bytes:
    """Generates a sleek dark-mode studio app icon PNG in memory using Pillow."""
    img = Image.new("RGBA", (size, size), (9, 13, 22, 255))
    draw = ImageDraw.Draw(img)

    # Outer subtle border ring
    pad = max(4, size // 16)
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=size // 5,
        fill=(17, 24, 39, 255),
        outline=(56, 189, 248, 220),
        width=max(3, size // 48),
    )

    # Audio waveform bars + play accent
    cx, cy = size // 2, size // 2
    bar_w = max(6, size // 18)
    gap = max(8, size // 12)
    heights = [0.22, 0.42, 0.58, 0.36, 0.24]
    offsets = [-2, -1, 0, 1, 2]
    for off, h_ratio in zip(offsets, heights):
        bx = cx + off * (bar_w + gap)
        bh = int(size * h_ratio)
        color = (56, 189, 248, 255) if off == 0 else (37, 99, 235, 255)
        draw.rounded_rectangle(
            [bx - bar_w // 2, cy - bh // 2, bx + bar_w // 2, cy + bh // 2],
            radius=bar_w // 2,
            fill=color,
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SERVICE_WORKER_JS = """
const CACHE_NAME = 'omnivoice-pwa-v2.25';
const ASSETS = ['/mobile', '/manifest.json', '/icon-192.png'];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS).catch(() => {}))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
"""


class TextPreviewRequest(BaseModel):
    text: str
    normalize_numbers: bool = False


MOBILE_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="OmniVoice">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#090d16">
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="192x192" href="/icon-192.png">
<title>OmniVoice Mobile</title>
<style>
  :root {
    --bg: #090d16;
    --surface: #111827;
    --elevated: #1e293b;
    --border: rgba(148, 163, 184, 0.16);
    --accent: #38bdf8;
    --btn-grad: linear-gradient(135deg, #2563eb 0%, #0284c7 100%);
    --text: #f8fafc;
    --muted: #94a3b8;
    --ok: #22c55e;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: calc(10px + env(safe-area-inset-top)) 12px calc(28px + env(safe-area-inset-bottom));
    -webkit-font-smoothing: antialiased;
  }
  .shell { max-width: 440px; margin: 0 auto; }

  /* Minimalist Top Bar */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 12px;
    background: rgba(17, 24, 39, 0.9);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 10px;
    position: sticky;
    top: 6px;
    z-index: 40;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 7px;
    font-weight: 800;
    font-size: 0.98rem;
    letter-spacing: -0.02em;
  }
  .dot-badge {
    font-size: 0.72rem;
    font-weight: 600;
    color: #86efac;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    padding: 2px 8px;
    border-radius: 999px;
  }
  .top-actions { display: flex; gap: 6px; align-items: center; }
  .icon-btn {
    color: var(--text);
    text-decoration: none;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 6px 10px;
    border-radius: 9px;
    border: 1px solid var(--border);
    background: var(--elevated);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .icon-btn:active { transform: scale(0.96); }

  /* Main Surface */
  .surface {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 10px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
  }
  .chips-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin-bottom: 8px;
  }
  .chips {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .chip {
    background: var(--elevated);
    border: 1px solid var(--border);
    color: #cbd5e1;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
    cursor: pointer;
    min-height: 28px;
  }
  .chip:active { transform: scale(0.96); border-color: var(--accent); color: #fff; }
  .char-count { font-size: 0.72rem; color: var(--muted); white-space: nowrap; }

  .textarea-wrap { position: relative; }
  textarea, select, input[type="text"], input[type="number"] {
    width: 100%;
    background: var(--elevated);
    border: 1px solid rgba(148, 163, 184, 0.24);
    color: var(--text);
    border-radius: 12px;
    padding: 11px 12px;
    font-size: 16px;
    line-height: 1.45;
    outline: none;
    font-family: inherit;
  }
  textarea:focus, select:focus, input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.18);
  }
  .clear-txt-btn {
    position: absolute;
    right: 8px;
    bottom: 8px;
    background: rgba(15, 23, 42, 0.75);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.7rem;
    padding: 2px 6px;
    cursor: pointer;
  }
  .row-2 {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 8px;
    margin-top: 10px;
  }
  .lbl {
    display: block;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .switch-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    background: var(--elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-top: 10px;
    font-size: 0.84rem;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-primary {
    width: 100%;
    min-height: 52px;
    margin-top: 12px;
    background: var(--btn-grad);
    color: #fff;
    border: 1px solid rgba(56, 189, 248, 0.45);
    border-radius: 13px;
    font-size: 0.98rem;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 6px 18px rgba(2, 132, 199, 0.3);
    transition: transform 0.1s ease;
  }
  .btn-primary:active { transform: scale(0.985); }
  .btn-primary:disabled { opacity: 0.65; cursor: wait; }

  /* Output Stage */
  .result-stage {
    display: none;
    background: var(--surface);
    border: 1px solid rgba(56, 189, 248, 0.38);
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 10px;
  }
  .status-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: #e2e8f0;
  }
  audio, video {
    width: 100%;
    border-radius: 10px;
    margin-top: 6px;
    background: #000;
  }
  .dl-row {
    display: flex;
    gap: 6px;
    margin-top: 8px;
  }
  .dl-link {
    flex: 1;
    text-align: center;
    background: var(--elevated);
    color: var(--accent);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 9px;
    padding: 8px 6px;
    font-size: 0.76rem;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
  }

  /* Progressive Disclosure Drawers */
  details.drawer {
    background: rgba(17, 24, 39, 0.72);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 8px;
    overflow: hidden;
  }
  details.drawer summary {
    list-style: none;
    padding: 11px 14px;
    font-size: 0.83rem;
    font-weight: 600;
    color: #cbd5e1;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
  }
  details.drawer summary::-webkit-details-marker { display: none; }
  details.drawer summary svg.chev { transition: transform 0.18s ease; color: var(--muted); }
  details.drawer[open] summary svg.chev { transform: rotate(180deg); color: var(--accent); }
  .drawer-body {
    padding: 12px 14px 14px;
    border-top: 1px solid var(--border);
  }
  .file-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    background: var(--elevated);
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 0.78rem;
  }
  /* Modal Sheet */
  .modal-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.72);
    backdrop-filter: blur(6px);
    z-index: 100;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }
  .modal-card {
    background: var(--surface);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 20px;
    max-width: 380px;
    width: 100%;
    padding: 18px;
  }
  .rec-btn {
    width: 100%;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid rgba(239, 68, 68, 0.45);
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    font-weight: 700;
    font-size: 0.82rem;
    cursor: pointer;
    margin-bottom: 8px;
  }
  .rec-btn.recording {
    background: #dc2626;
    color: #fff;
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition: none !important; animation: none !important; }
  }
</style>
</head>
<body>
<div class="shell">
  <!-- Minimalist Header -->
  <header class="topbar">
    <div class="brand">
      <span>🎙️ OmniVoice</span>
      <span class="dot-badge" id="hdr-badge">● Ready</span>
    </div>
    <div class="top-actions">
      <button type="button" class="icon-btn" onclick="openInstallModal()" title="Kết nối Điện Thoại & Cài App">
        <span>📲 Cài App / Link</span>
      </button>
      <a class="icon-btn" href="/" title="Mở giao diện Studio đầy đủ">PC &rarr;</a>
    </div>
  </header>

  <!-- Primary Composer Card -->
  <main class="surface">
    <div class="chips-bar">
      <div class="chips">
        <button type="button" class="chip" onclick="setTpl('fast')">⚡ Mẫu 2.5s</button>
        <button type="button" class="chip" onclick="setTpl('news')">📰 Tin 30s</button>
        <button type="button" class="chip" onclick="setTpl('story')">🌙 Truyện</button>
        <button type="button" class="chip" onclick="previewCleanText()" title="Lọc ký tự Markdown">🧹 Lọc chữ</button>
      </div>
      <span class="char-count" id="char-counter">45 ký tự</span>
    </div>

    <div class="textarea-wrap">
      <textarea id="tts-text" rows="3" oninput="updateCount()" aria-label="Kịch bản văn bản" placeholder="Nhập nội dung cần tạo giọng nói & video...">Chào bạn, hôm nay thời tiết Cần Thơ rất đẹp.</textarea>
      <button type="button" class="clear-txt-btn" onclick="clearText()" title="Xóa nhanh">✕</button>
    </div>

    <div class="row-2">
      <div>
        <label class="lbl" for="tts-voice">Giọng đọc</label>
        <select id="tts-voice"></select>
      </div>
      <div>
        <label class="lbl" for="tts-mode">Tốc độ AI</label>
        <select id="tts-mode">
          <option value="TURBO" selected>TURBO (4-Step)</option>
          <option value="FAST">FAST (8-Step)</option>
          <option value="BALANCED">BALANCED</option>
          <option value="QUALITY">QUALITY</option>
        </select>
      </div>
    </div>

    <label class="switch-line" for="tts-make-video">
      <span>🎬 Tự động xuất kèm Video MP4</span>
      <input type="checkbox" id="tts-make-video" checked style="width:20px;height:20px;accent-color:#38bdf8;">
    </label>

    <button type="button" id="btn-run" class="btn-primary" onclick="runGenerate()">
      🚀 Tạo Giọng Đọc & Video
    </button>
  </main>

  <!-- Unified Output Stage -->
  <section class="result-stage" id="result-card" aria-live="polite">
    <div class="status-strip">
      <span id="res-summary">✅ Hoàn tất</span>
      <span id="res-badge" class="dot-badge">⚡ CACHE</span>
    </div>
    <div id="video-wrap" style="display:none;">
      <video id="video-player" controls playsinline preload="metadata"></video>
    </div>
    <audio id="audio-player" controls preload="metadata"></audio>
    <div class="dl-row">
      <a id="audio-dl" class="dl-link" href="#" download>⬇️ Tải WAV</a>
      <a id="video-dl" class="dl-link" href="#" download style="display:none;">⬇️ Tải MP4</a>
      <button type="button" id="share-btn" class="dl-link" onclick="shareOutput()" style="display:none;">📤 Chia sẻ</button>
    </div>
  </section>

  <!-- Progressive Drawer 1: Full Advanced Audio & Video Controls -->
  <details class="drawer">
    <summary>
      <span>🎛️ Tùy chỉnh nâng cao (Giọng nói & Video)</span>
      <svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
    </summary>
    <div class="drawer-body">
      <div class="row-2">
        <div>
          <label class="lbl" for="adv-speed">Tốc độ nói (0.8 - 1.3)</label>
          <input type="number" id="adv-speed" value="1.0" step="0.05" min="0.8" max="1.3">
        </div>
        <div>
          <label class="lbl" for="adv-lang">Ngôn ngữ</label>
          <select id="adv-lang">
            <option value="Vietnamese" selected>Tiếng Việt</option>
            <option value="Auto">Tự động</option>
            <option value="English">English</option>
            <option value="Chinese">Chinese</option>
            <option value="Japanese">Japanese</option>
          </select>
        </div>
      </div>

      <label class="switch-line" for="adv-norm-num" style="margin-top:8px;">
        <span>🔢 Tự động đọc số thành chữ tiếng Việt</span>
        <input type="checkbox" id="adv-norm-num" style="width:18px;height:18px;accent-color:#38bdf8;">
      </label>

      <div class="row-2">
        <div>
          <label class="lbl" for="adv-preset">Chế độ Encode MP4</label>
          <select id="adv-preset">
            <option value="TURBO_CACHED_MUX" selected>Cached Mux (3s)</option>
            <option value="TURBO_DIRECT_ENCODE">Direct AVX2 (15s)</option>
            <option value="AMD_AMF_HW">AMD Vega 8 HW</option>
            <option value="MEDIAFOUNDATION_HW">MediaFoundation</option>
            <option value="BASELINE_SW">Baseline SW</option>
          </select>
        </div>
        <div>
          <label class="lbl" for="adv-res">Độ phân giải & FPS</label>
          <select id="adv-res">
            <option value="1280x720|24" selected>720p @ 24fps</option>
            <option value="1280x720|30">720p @ 30fps</option>
            <option value="1920x1080|24">1080p @ 24fps</option>
            <option value="1920x1080|30">1080p @ 30fps</option>
          </select>
        </div>
      </div>

      <div class="row-2">
        <div>
          <label class="lbl" for="adv-title">Tiêu đề trên Video</label>
          <input type="text" id="adv-title" value="OmniVoice Studio">
        </div>
        <div>
          <label class="lbl" for="adv-sub">Phụ đề Video</label>
          <input type="text" id="adv-sub" value="AI Voice & Video">
        </div>
      </div>

      <div style="margin-top:10px;">
        <label class="lbl" for="adv-bg-file">🖼️ Ảnh nền Video tùy chỉnh (Từ thư viện ảnh ĐT)</label>
        <input type="file" id="adv-bg-file" accept="image/*" style="font-size:0.78rem;color:var(--muted);">
      </div>

      <div style="margin-top:8px;">
        <label class="lbl" for="adv-wav-file">🔊 Hoặc chọn file Audio (.wav) có sẵn để ghép Video ngay</label>
        <input type="file" id="adv-wav-file" accept="audio/*" style="font-size:0.78rem;color:var(--muted);">
      </div>

      <div style="margin-top:8px;">
        <label class="lbl" for="adv-oneshot-file">⚡ Clone nhanh 1-chạm (Chọn mẫu 3-8s không cần lưu)</label>
        <input type="file" id="adv-oneshot-file" accept="audio/*" style="font-size:0.78rem;color:var(--muted);">
      </div>
    </div>
  </details>

  <!-- Progressive Drawer 2: Voice Cloning & Library Management -->
  <details class="drawer">
    <summary>
      <span>🎙️ Thu âm & Quản lý Giọng mẫu (Voice Clone)</span>
      <svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
    </summary>
    <div class="drawer-body">
      <label class="lbl" for="clone-name">Tên hồ sơ giọng mới</label>
      <input type="text" id="clone-name" placeholder="VD: Giọng Của Tôi" style="margin-bottom:8px;">

      <button type="button" id="btn-mic-rec" class="rec-btn" onclick="toggleMicRecording()">
        ⏺️ Bấm để Thu âm trực tiếp từ Micro (3–8 giây)
      </button>
      <audio id="rec-preview" controls style="display:none;margin-bottom:8px;height:36px;"></audio>

      <label class="lbl" for="clone-file">Hoặc chọn file ghi âm có sẵn trên máy</label>
      <input type="file" id="clone-file" accept="audio/*" capture="user" style="font-size:0.78rem;color:var(--muted);margin-bottom:8px;">

      <label class="lbl" for="clone-text">Câu nói trong mẫu thu âm</label>
      <input type="text" id="clone-text" value="Xin chào các bạn, hôm nay trời rất đẹp.">

      <div style="display:flex;gap:6px;margin-top:10px;">
        <button type="button" class="chip" style="flex:2;padding:9px;border-radius:10px;color:#fff;border-color:var(--accent);" onclick="cloneVoice()">💾 Lưu Giọng Mới (.pt)</button>
        <button type="button" class="chip" style="flex:1;padding:9px;border-radius:10px;color:#fca5a5;" onclick="deleteCurrentVoice()">🗑️ Xóa giọng đang chọn</button>
      </div>
      <div id="clone-msg" style="font-size:0.76rem;margin-top:6px;color:#86efac;"></div>
    </div>
  </details>

  <!-- Progressive Drawer 3: Output Media Gallery & System Info -->
  <details class="drawer" ontoggle="if(this.open) loadInfo()">
    <summary>
      <span>📂 Kho Video Đã Xuất & Hệ Thống</span>
      <svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
    </summary>
    <div class="drawer-body">
      <div id="sys-summary" style="font-size:0.75rem;color:var(--muted);margin-bottom:8px;line-height:1.45;"></div>
      <div id="gallery-list" style="font-size:0.78rem;color:var(--muted);">Đang tải...</div>
    </div>
  </details>
</div>

<!-- Install / Connection Modal -->
<div class="modal-backdrop" id="install-modal" onclick="if(event.target===this) closeInstallModal()">
  <div class="modal-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-weight:800;font-size:0.95rem;">📲 Kết Nối & Cài Đặt App Điện Thoại</span>
      <button type="button" class="chip" onclick="closeInstallModal()">Đóng</button>
    </div>
    <div id="pwa-install-btn-wrap" style="display:none;margin-bottom:10px;">
      <button type="button" class="btn-primary" style="margin-top:0;min-height:44px;" onclick="triggerPwaInstall()">
        📲 Cài Đặt App Ra Màn Hình Chính Ngay
      </button>
    </div>
    <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5;">
      <div style="margin-bottom:8px;"><b>1. Đường dẫn truy cập trên Điện Thoại:</b></div>
      <div id="modal-links" style="background:var(--elevated);padding:10px;border-radius:10px;border:1px solid var(--border);font-family:monospace;font-size:0.8rem;margin-bottom:10px;word-break:break-all;"></div>
      <div><b>2. Cách thêm ra Màn hình chính (Chạy Full-Screen):</b><br>
      • <b>iPhone (Safari):</b> Bấm nút <b>Chia sẻ</b> ở thanh dưới &rarr; <b>Thêm vào MH chính</b>.<br>
      • <b>Android (Chrome):</b> Bấm <b>Menu 3 chấm</b> &rarr; <b>Cài đặt ứng dụng / Thêm vào MH chính</b>.</div>
    </div>
  </div>
</div>

<script>
// Register PWA Service Worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

let deferredPrompt = null;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById('pwa-install-btn-wrap').style.display = 'block';
});

async function triggerPwaInstall() {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  document.getElementById('pwa-install-btn-wrap').style.display = 'none';
}

function openInstallModal() {
  loadInfo();
  document.getElementById('install-modal').style.display = 'flex';
}
function closeInstallModal() {
  document.getElementById('install-modal').style.display = 'none';
}

const TPLS = {
  fast: "Chào bạn, hôm nay thời tiết Cần Thơ rất đẹp.",
  news: "Chào mừng các bạn đến với hệ thống sản xuất giọng nói và video tự động OmniVoice. Phiên bản mới nhất đã được tối ưu hóa riêng cho máy tính Dell Inspiron và điện thoại di động.",
  story: "Có những người xuất hiện đúng lúc trái tim muốn yêu, nhưng lại sai lúc cả hai biết cách ở bên nhau."
};

function updateCount() {
  const len = document.getElementById('tts-text').value.length;
  document.getElementById('char-counter').textContent = len + " ký tự";
}
function setTpl(k) {
  document.getElementById('tts-text').value = TPLS[k];
  updateCount();
}
function clearText() {
  document.getElementById('tts-text').value = "";
  updateCount();
  document.getElementById('tts-text').focus();
}

async function previewCleanText() {
  const txt = document.getElementById('tts-text').value.trim();
  if (!txt) return;
  const norm = document.getElementById('adv-norm-num').checked;
  const r = await fetch('/api/mobile/preview_text', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: txt, normalize_numbers: norm})
  });
  const d = await r.json();
  if (d.cleaned_text) {
    document.getElementById('tts-text').value = d.cleaned_text;
    updateCount();
  }
}

let lastMediaUrl = null;
let lastMediaTitle = "OmniVoice Video";

async function shareOutput() {
  if (navigator.share && lastMediaUrl) {
    try {
      await navigator.share({ title: lastMediaTitle, url: window.location.origin + lastMediaUrl });
    } catch (e) {}
  }
}

function playInStage(filename) {
  const url = "/api/mobile/media/" + encodeURIComponent(filename) + "?t=" + Date.now();
  document.getElementById('result-card').style.display = 'block';
  document.getElementById('res-summary').textContent = "▶️ Đang phát: " + filename;
  if (filename.toLowerCase().endsWith('.mp4')) {
    document.getElementById('video-wrap').style.display = 'block';
    document.getElementById('video-player').src = url;
    document.getElementById('video-player').play().catch(()=>{});
    document.getElementById('video-dl').style.display = 'inline-block';
    document.getElementById('video-dl').href = url;
  } else {
    document.getElementById('audio-player').src = url;
    document.getElementById('audio-player').play().catch(()=>{});
    document.getElementById('audio-dl').href = url;
  }
  document.getElementById('result-card').scrollIntoView({behavior: 'smooth'});
}

async function loadInfo() {
  try {
    const r = await fetch('/api/mobile/info');
    const data = await r.json();
    const sel = document.getElementById('tts-voice');
    const currentVal = sel.value;
    sel.innerHTML = '';
    data.voices.forEach((v, idx) => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      if (currentVal ? v === currentVal : idx === 0) opt.selected = true;
      sel.appendChild(opt);
    });

    document.getElementById('hdr-badge').textContent = `● D: ${data.d_free_gb}GB`;
    document.getElementById('sys-summary').innerHTML =
      `<b>Thiết bị:</b> ${data.cpu} (${data.threads} Threads MKL) · <b>RAM:</b> ${data.ram_mb} MB · <b>Ổ C:</b> trống ${data.c_free_gb}GB (Đã khóa ghi tạm) · <b>Ổ D:</b> trống ${data.d_free_gb}GB`;

    let linksHtml = '';
    if (data.public_url) {
      linksHtml += `<div style="margin-bottom:6px;color:#86efac;">🌐 <b>Public HTTPS (Mọi mạng 4G/Wi-Fi):</b><br><a href="${data.public_url}/mobile" style="color:#38bdf8;">${data.public_url}/mobile</a></div>`;
    }
    linksHtml += data.lan_urls.map(u => `<div>📶 <b>Wi-Fi nội bộ:</b> <a href="${u}/mobile" style="color:#38bdf8;">${u}/mobile</a></div>`).join('');
    document.getElementById('modal-links').innerHTML = linksHtml;

    const gList = document.getElementById('gallery-list');
    if (data.recent_files && data.recent_files.length > 0) {
      gList.innerHTML = data.recent_files.map(f => `
        <div class="file-item">
          <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:58%;">${f.name} <small style="color:#64748b;">(${f.size_mb}MB)</small></span>
          <div style="display:flex;gap:8px;">
            <a href="javascript:void(0)" onclick="playInStage('${f.name}')" style="color:#86efac;font-weight:700;text-decoration:none;">Phát</a>
            <a href="/api/mobile/media/${encodeURIComponent(f.name)}" download style="color:#38bdf8;font-weight:700;text-decoration:none;">Tải</a>
          </div>
        </div>
      `).join('');
    } else {
      gList.textContent = "Chưa có file nào.";
    }
  } catch (e) { console.error(e); }
}

async function runGenerate() {
  const text = document.getElementById('tts-text').value.trim();
  const wavFile = document.getElementById('adv-wav-file').files[0];
  if (!text && !wavFile) {
    alert("Vui lòng nhập kịch bản hoặc chọn file Audio có sẵn!");
    return;
  }
  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  const t0 = performance.now();
  const timer = setInterval(() => {
    const el = ((performance.now() - t0) / 1000).toFixed(1);
    btn.textContent = `⏳ Đang xử lý... (${el}s)`;
  }, 150);

  try {
    const resParts = document.getElementById('adv-res').value.split('|');
    const fd = new FormData();
    fd.append("text", text);
    fd.append("voice", document.getElementById('tts-voice').value);
    fd.append("mode", document.getElementById('tts-mode').value);
    fd.append("make_video", document.getElementById('tts-make-video').checked ? "true" : "false");
    fd.append("speed", document.getElementById('adv-speed').value || "1.0");
    fd.append("language", document.getElementById('adv-lang').value);
    fd.append("normalize_numbers", document.getElementById('adv-norm-num').checked ? "true" : "false");
    fd.append("video_preset", document.getElementById('adv-preset').value);
    fd.append("resolution", resParts[0] || "1280x720");
    fd.append("fps", resParts[1] || "24");
    fd.append("title", document.getElementById('adv-title').value || "OmniVoice Studio");
    fd.append("subtitle", document.getElementById('adv-sub').value || "AI Voice & Video");

    const bgFile = document.getElementById('adv-bg-file').files[0];
    if (bgFile) fd.append("bg_image", bgFile);
    if (wavFile) fd.append("existing_wav", wavFile);
    const oneshotFile = document.getElementById('adv-oneshot-file').files[0];
    if (oneshotFile) fd.append("oneshot_audio", oneshotFile);

    const r = await fetch('/api/mobile/generate_full', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Lỗi xử lý");

    document.getElementById('result-card').style.display = 'block';
    document.getElementById('res-summary').textContent =
      `✅ ${data.total_time_sec}s · Thời lượng ${data.audio_duration_sec}s` + (data.video_fps ? ` · ${data.video_fps} FPS` : '');
    document.getElementById('res-badge').textContent = data.cached_hit ? "⚡ CACHE" : "🧠 CPU";

    const audUrl = "/api/mobile/media/" + encodeURIComponent(data.audio_filename) + "?t=" + Date.now();
    document.getElementById('audio-player').src = audUrl;
    document.getElementById('audio-dl').href = audUrl;
    lastMediaUrl = audUrl;

    const vidWrap = document.getElementById('video-wrap');
    const vidDl = document.getElementById('video-dl');
    if (data.video_filename) {
      vidWrap.style.display = 'block';
      vidDl.style.display = 'inline-block';
      const vidUrl = "/api/mobile/media/" + encodeURIComponent(data.video_filename) + "?t=" + Date.now();
      document.getElementById('video-player').src = vidUrl;
      vidDl.href = vidUrl;
      lastMediaUrl = vidUrl;
    } else {
      vidWrap.style.display = 'none';
      vidDl.style.display = 'none';
    }

    if (navigator.share) {
      document.getElementById('share-btn').style.display = 'inline-block';
    }
    document.getElementById('result-card').scrollIntoView({behavior: 'smooth'});
  } catch (err) {
    alert("Lỗi: " + err.message);
  } finally {
    clearInterval(timer);
    btn.disabled = false;
    btn.textContent = "🚀 Tạo Giọng Đọc & Video";
  }
}

// Built-in WebAudio WAV Microphone Recorder
let mediaStream = null;
let audioCtx = null;
let processor = null;
let recordedChunks = [];
let recordedWavBlob = null;
let isRecording = false;

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

async function toggleMicRecording() {
  const btn = document.getElementById('btn-mic-rec');
  const msg = document.getElementById('clone-msg');
  if (!isRecording) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      msg.textContent = "ℹ️ Trình duyệt chặn Micro qua HTTP. Hãy bấm 'Chọn file' bên dưới hoặc dùng link HTTPS.";
      document.getElementById('clone-file').click();
      return;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      recordedChunks = [];
      processor.onaudioprocess = (e) => {
        if (isRecording) {
          recordedChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        }
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      isRecording = true;
      btn.classList.add('recording');
      btn.textContent = "⏹️ Đang thu âm... (Bấm để Dừng)";
    } catch (err) {
      msg.textContent = "ℹ️ Hãy chọn file thu âm bên dưới (" + err.message + ")";
      document.getElementById('clone-file').click();
    }
  } else {
    isRecording = false;
    btn.classList.remove('recording');
    btn.textContent = "⏺️ Thu âm lại từ Micro";
    if (processor) processor.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    const sr = audioCtx ? audioCtx.sampleRate : 44100;
    let totalLen = recordedChunks.reduce((acc, c) => acc + c.length, 0);
    let merged = new Float32Array(totalLen);
    let offset = 0;
    for (let c of recordedChunks) {
      merged.set(c, offset);
      offset += c.length;
    }
    recordedWavBlob = encodeWAV(merged, sr);
    const prev = document.getElementById('rec-preview');
    prev.src = URL.createObjectURL(recordedWavBlob);
    prev.style.display = 'block';
    msg.textContent = "✅ Đã thu âm xong! Nhập tên giọng và bấm Lưu.";
  }
}

async function cloneVoice() {
  const name = document.getElementById('clone-name').value.trim();
  const fileInput = document.getElementById('clone-file');
  const text = document.getElementById('clone-text').value.trim();
  const msg = document.getElementById('clone-msg');
  const audioSource = fileInput.files[0] || recordedWavBlob;
  if (!name || !audioSource) {
    msg.textContent = "⚠️ Nhập tên giọng và thu âm/chọn file mẫu (3-8s).";
    return;
  }
  msg.textContent = "⏳ Đang mã hóa giọng nói sang prompt.pt...";
  const fd = new FormData();
  fd.append("name", name);
  fd.append("ref_text", text || "Xin chào các bạn.");
  fd.append("audio", audioSource, fileInput.files[0] ? fileInput.files[0].name : "mic_record.wav");
  try {
    const r = await fetch('/api/mobile/clone_voice', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Lỗi lưu giọng");
    msg.textContent = `✅ Đã lưu giọng '${data.voice}'!`;
    await loadInfo();
    document.getElementById('tts-voice').value = data.voice;
  } catch (e) {
    msg.textContent = "❌ " + e.message;
  }
}

async function deleteCurrentVoice() {
  const voice = document.getElementById('tts-voice').value;
  const msg = document.getElementById('clone-msg');
  if (!voice || voice === "[Giọng Mặc Định]") {
    msg.textContent = "⚠️ Không thể xóa giọng mặc định.";
    return;
  }
  if (!confirm(`Xóa giọng '${voice}' khỏi thư viện?`)) return;
  const r = await fetch('/api/mobile/voices/' + encodeURIComponent(voice), { method: 'DELETE' });
  if (r.ok) {
    msg.textContent = `🗑️ Đã xóa giọng '${voice}'.`;
    await loadInfo();
  }
}

document.addEventListener('DOMContentLoaded', loadInfo);
</script>
</body>
</html>
"""


def attach_mobile_pwa_routes(app: FastAPI, engine: OmniVoiceEngine, port: int = 8001) -> None:
    """Attaches `/mobile`, `/sw.js`, `/manifest.json`, icons, and `/api/mobile/*` routes."""

    icon_192_bytes = _generate_app_icon_png(192)
    icon_512_bytes = _generate_app_icon_png(512)

    @app.get("/icon-192.png")
    @app.get("/apple-touch-icon.png")
    async def get_icon_192():
        return Response(content=icon_192_bytes, media_type="image/png")

    @app.get("/icon-512.png")
    async def get_icon_512():
        return Response(content=icon_512_bytes, media_type="image/png")

    @app.get("/sw.js")
    async def get_service_worker():
        return Response(content=SERVICE_WORKER_JS, media_type="application/javascript")

    @app.get("/manifest.json")
    async def pwa_manifest():
        return JSONResponse({
            "name": "OmniVoice AI Voice & Video Studio",
            "short_name": "OmniVoice",
            "description": "Zero-Shot Voice Cloning & Fast Video Export Studio",
            "start_url": "/mobile",
            "scope": "/",
            "display": "standalone",
            "background_color": "#090d16",
            "theme_color": "#090d16",
            "orientation": "portrait",
            "icons": [
                {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
            ],
        })

    @app.get("/mobile", response_class=HTMLResponse)
    @app.get("/m", response_class=HTMLResponse)
    async def mobile_app_page():
        return HTMLResponse(content=MOBILE_HTML)

    @app.get("/api/mobile/info")
    async def mobile_info():
        voices = engine.voice_manager.list_voices()
        voices = (voices + ["[Giọng Mặc Định]"]) if voices else ["[Giọng Mặc Định]"]
        ips = get_lan_ips()
        lan_urls = [f"http://{ip}:{port}" for ip in ips]
        res_info = get_detailed_resource_usage(str(engine.storage.base_dir))

        out_dir = engine.storage.get_outputs_dir()
        recent = []
        if out_dir.exists():
            files = sorted(
                [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in (".mp4", ".wav")],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )[:12]
            for f in files:
                recent.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                })

        return {
            "voices": voices,
            "cpu": engine.hw_info.get("cpu_name", "AMD Ryzen 5 3450U"),
            "threads": engine.hw_info.get("recommended_threads", 6),
            "ram_mb": res_info.get("ram_process_mb", 0),
            "c_free_gb": res_info.get("c_drive_free_gb", 0),
            "d_free_gb": res_info.get("d_drive_free_gb", 0),
            "lan_urls": lan_urls,
            "public_url": PUBLIC_TUNNEL_URL,
            "recent_files": recent,
        }

    @app.post("/api/mobile/preview_text")
    async def mobile_preview_text(req: TextPreviewRequest):
        cleaned = TextPreprocessor.process(req.text, normalize_nums=req.normalize_numbers)
        return {"cleaned_text": cleaned}

    @app.post("/api/mobile/generate_full")
    async def mobile_generate_full(
        text: str = Form(""),
        voice: str = Form(""),
        mode: str = Form("TURBO"),
        make_video: str = Form("true"),
        speed: float = Form(1.0),
        language: str = Form("Vietnamese"),
        normalize_numbers: str = Form("false"),
        video_preset: str = Form("TURBO_CACHED_MUX"),
        resolution: str = Form("1280x720"),
        fps: int = Form(24),
        title: str = Form("OmniVoice Studio"),
        subtitle: str = Form("AI Voice & Video"),
        bg_image: Optional[UploadFile] = File(None),
        existing_wav: Optional[UploadFile] = File(None),
        oneshot_audio: Optional[UploadFile] = File(None),
    ):
        t0 = time.perf_counter()
        do_video = make_video.lower() in ("true", "1", "yes", "on")
        do_norm = normalize_numbers.lower() in ("true", "1", "yes", "on")
        voice_arg = None if voice in (None, "", "[Giọng Mặc Định]") else voice

        temp_dir = engine.storage.get_temp_dir()
        out_dir = engine.storage.get_outputs_dir()

        bg_path = None
        if bg_image and bg_image.filename:
            bg_path = str(temp_dir / f"mob_bg_{int(time.time())}_{Path(bg_image.filename).name}")
            with open(bg_path, "wb") as f:
                shutil.copyfileobj(bg_image.file, f)

        oneshot_path = None
        if oneshot_audio and oneshot_audio.filename:
            oneshot_path = str(temp_dir / f"mob_oneshot_{int(time.time())}_{Path(oneshot_audio.filename).name}")
            with open(oneshot_path, "wb") as f:
                shutil.copyfileobj(oneshot_audio.file, f)

        if existing_wav and existing_wav.filename:
            audio_path = out_dir / f"mob_input_{int(time.time())}_{Path(existing_wav.filename).name}"
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(existing_wav.file, f)
            tts_time = 0.0
            audio_dur = 0.0
            cached_hit = True
        else:
            if not text or not text.strip():
                raise HTTPException(status_code=400, detail="Vui lòng nhập kịch bản văn bản")
            res = engine.synthesize(
                text=text.strip(),
                voice_name=voice_arg,
                ref_audio_path=oneshot_path,
                language=language,
                perf_mode=mode,
                speed=float(speed),
                normalize_numbers=do_norm,
            )
            tts_time = res.generation_time
            audio_dur = res.audio_duration
            cached_hit = bool(res.cached_hit)
            audio_path = Path(res.output_path) if res.output_path else (out_dir / f"mob_audio_{int(time.time())}.wav")
            if not audio_path.exists():
                sf.write(str(audio_path), res.audio, res.sample_rate)

        video_filename = None
        video_fps = 0.0
        if do_video:
            vres = engine.export_video(
                text=text.strip() or "OmniVoice Mobile Video",
                audio_path=str(audio_path),
                voice_name=voice_arg,
                perf_mode=mode,
                video_preset=video_preset or "TURBO_CACHED_MUX",
                resolution=resolution or "1280x720",
                fps=int(fps or 24),
                title=title or "OmniVoice Studio",
                subtitle=subtitle or f"Voice: {voice or 'Default'} | Mode: {mode}",
                custom_image_path=bg_path,
                use_audio_cache=True,
                use_visual_cache=True,
            )
            video_filename = Path(vres.output_video_path).name
            video_fps = round(vres.rendering_fps, 1)
            if audio_dur <= 0:
                audio_dur = vres.video_duration_sec

        total_sec = round(time.perf_counter() - t0, 3)
        return {
            "total_time_sec": total_sec,
            "audio_duration_sec": round(audio_dur, 2),
            "tts_time_sec": round(tts_time, 3),
            "cached_hit": cached_hit,
            "audio_filename": audio_path.name,
            "video_filename": video_filename,
            "video_fps": video_fps,
        }

    @app.post("/api/mobile/clone_voice")
    async def mobile_clone_voice(
        name: str = Form(...),
        ref_text: str = Form("Xin chào các bạn."),
        audio: UploadFile = File(...),
    ):
        tmp_audio = engine.storage.get_temp_dir() / f"upload_clone_{int(time.time())}_{Path(audio.filename or 'ref.wav').name}"
        with open(tmp_audio, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        profile = engine.voice_manager.create_or_update_voice(
            name=name.strip(),
            ref_audio_path=str(tmp_audio),
            ref_text=ref_text.strip() or "Xin chào các bạn.",
            model=engine.model,
            description="Cloned via Mobile PWA",
        )
        return {"voice": profile.name}

    @app.delete("/api/mobile/voices/{voice_name}")
    async def mobile_delete_voice(voice_name: str):
        ok = engine.voice_manager.delete_voice(voice_name.strip())
        if not ok:
            raise HTTPException(status_code=404, detail="Không tìm thấy giọng")
        return {"deleted": voice_name}

    @app.get("/api/mobile/media/{filename}")
    async def mobile_media(filename: str):
        safe_name = Path(filename).name
        for c in (engine.storage.get_outputs_dir() / safe_name, engine.storage.get_temp_dir() / safe_name):
            if c.exists() and c.is_file():
                media_type = "video/mp4" if c.suffix.lower() == ".mp4" else "audio/wav"
                return FileResponse(str(c), media_type=media_type, filename=safe_name)
        raise HTTPException(status_code=404, detail="File not found")
