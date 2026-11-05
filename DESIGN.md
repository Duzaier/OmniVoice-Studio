# OmniVoice Design System (`DESIGN.md`) — V2.24 Swiss Minimalist Edition

## 1. Brand Identity & Personality Archetype
- **Archetype:** *Swiss Minimalist Audio/Video Studio* (inspired by Linear, Teenage Engineering, and Apple Logic Remote).
- **Core Philosophy:** **Zero Duplication & Radical Progressive Disclosure** (*"Cái gì không cần hiện thì ẩn đi hết"*).
- **Focal Point:** The Script Composer and the Single Primary Action (`Tạo Giọng & Video`). All secondary parameters (emotion curves, pause scales, FFmpeg codec presets, custom slide backgrounds, voice cloning, and hardware telemetry) remain tucked inside on-demand drawers until explicitly opened.

## 2. Color Palette (Semantic Tokens & WCAG 2.2 AA Contrast)
| Token | Value | Contrast vs `#090d16` | Usage |
| :--- | :--- | :--- | :--- |
| `--ov-bg` | `#090d16` | — | Root canvas background |
| `--ov-surface` | `#111827` | 1.2:1 | Primary studio card container |
| `--ov-surface-elevated` | `#1e293b` | 1.7:1 | Inputs, drawers, chips |
| `--ov-border` | `rgba(148, 163, 184, 0.16)` | — | Hairline structural borders |
| `--ov-accent` | `#38bdf8` | 9.4:1 (AAA) | Active focus ring, links, status highlights |
| `--ov-primary-btn` | `linear-gradient(135deg, #2563eb, #0284c7)` | 5.2:1 (AA) | Primary action trigger |
| `--ov-text` | `#f8fafc` | 17.8:1 (AAA) | Primary labels & input text |
| `--ov-muted` | `#94a3b8` | 6.8:1 (AA) | Secondary hints & collapsed summaries |

## 3. Typography Hierarchy
- **Font Stack:** `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- **Title Scale:** `clamp(1.05rem, 1.5vw, 1.25rem)`, `font-weight: 700`, `letter-spacing: -0.02em`
- **Input Text:** `16px` (`1rem`) minimum floor to prevent iOS Safari / Android Chrome viewport auto-zoom on focus.
- **Micro Labels / Chips:** `12.5px` (`0.78rem`), `font-weight: 600` (strictly >= 12px floor).

## 4. Spacing & Grid System
- **Base Unit:** `4px` / `8px` rhythm (`8px`, `12px`, `16px`, `24px`).
- **Desktop Container:** Max-width `960px` centered (2-column asymmetric split: `55%` Input Composer / `45%` Output Stage).
- **Mobile / Preview Container:** Max-width `440px` single-column stack with sticky bottom navigation.

## 5. Component Language & Uiverse / Morphicons / GSAP Rules
- **Buttons (`uiverse-ui-engineering`):** Scoped `.ov-btn-primary` (`min-height: 52px`, tactile `:active` scale `0.985`, high-contrast `:focus-visible` ring).
- **Progressive Drawers:** Scoped `<details>` / Gradio `Accordion(open=False)` with smooth height/opacity transitions.
- **Icon Transitions (`morphicons-icon-motion`):** Stroke-based SVG state morphing (`ChevronDown ↔ ChevronUp` on drawer expand, `Monitor ↔ Smartphone` on viewport toggle) with `prefers-reduced-motion: reduce` fallback.
- **Motion (`gsap-core`):** Entrance and drawer transitions use `autoAlpha` and `y` transforms inside `gsap.matchMedia()` so `prefers-reduced-motion: reduce` users get instant `duration: 0` state updates.

## 6. Anti-Patterns Eliminated in V2.24
- ❌ **No Duplicate Tabs:** Merged the 3 separate TTS / Video / Mobile tabs into **1 Unified Studio** + **1 Voice Cloning Drawer** + **1 System Info Popover**.
- ❌ **No Persistent Banner Soup:** Removed the multi-line yellow/green hardware and disk warning boxes from the top of the page; replaced with a single `32px` live status pill (`● 6-Thread MKL · D: 300GB`).
- ❌ **No Exposed Secondary Sliders by Default:** Speed, pause scale, emotion, number normalization, FFmpeg encoder preset, resolution, FPS, and custom backgrounds are hidden inside **`🎛️ Tùy chỉnh nâng cao`** until needed.
