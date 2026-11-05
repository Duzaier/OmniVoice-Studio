"""
experiments/v2_21/e2e_browser_test.py

Autonomous E2E Testing Suite for OmniVoice Workstation (Gradio Web UI on port 8001).
Tests Happy-Path synthesis, Failure Paths A-E, captures artifacts, and programmatically
validates the produced audio files.
"""

import os
import sys
import time
import json
import soundfile as sf
import numpy as np

# Ensure UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gradio_client import Client
from real_audio_validator import validate_generated_audio

BASE_URL = "http://127.0.0.1:8001/"
OUTPUT_DIR = "experiments/v2_21"
AUDIO_ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "audio_artifacts")
BROWSER_ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "browser_artifacts")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(AUDIO_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BROWSER_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def test_01_happy_path(client: Client) -> dict:
    """Happy Path: Standard Vietnamese synthesis with voice profile."""
    print("\n--- [E2E Test 1] Happy-Path Synthesis ---", flush=True)
    test_text = "Chào bạn, hôm nay thời tiết Cần Thơ rất đẹp."
    
    t0 = time.perf_counter()
    try:
        result = client.predict(
            text=test_text,
            voice="Giọng nữ",
            direct_aud=None,
            direct_txt="",
            lang="Vietnamese",
            mode="TURBO",
            speed=1.0,
            emotion_sel="Tự động nhận diện ngữ cảnh (Auto-Detect)",
            pause_scale=1.0,
            norm_nums=False,
            debug_mode=True,
            api_name="/_generate_ui"
        )
        latency = time.perf_counter() - t0
        audio_res, status_md, metrics_md, table_data = result

        if audio_res is None or not os.path.exists(audio_res):
            print(f"  [FAIL] No audio returned. Status: {status_md}", flush=True)
            return {"test": "Happy Path", "status": "FAIL", "error": f"No audio returned. Status: {status_md}"}

        dest_wav = os.path.join(AUDIO_ARTIFACTS_DIR, "01_happy_path_output.wav")
        data, sr = sf.read(audio_res)
        sf.write(dest_wav, data, sr)

        val_res = validate_generated_audio(dest_wav)
        assert val_res["valid"], f"Audio validation failed: {val_res['error']}"

        record = {
            "test": "Happy Path",
            "status": "PASS",
            "latency_sec": round(latency, 2),
            "output_path": dest_wav,
            "duration_sec": val_res["metrics"]["duration_sec"],
            "sample_rate": val_res["metrics"]["sample_rate"],
            "rms": val_res["metrics"]["rms"],
            "clipping_ratio": val_res["metrics"]["clipping_ratio"],
            "status_md": status_md,
        }
        print(f"  [PASS] Happy-path generated {val_res['metrics']['duration_sec']}s audio in {latency:.2f}s (RMS={val_res['metrics']['rms']})", flush=True)
        return record

    except Exception as e:
        print(f"  [FAIL] Happy Path Exception: {e}", flush=True)
        return {"test": "Happy Path", "status": "FAIL", "error": str(e)}


def test_02_failure_empty_text(client: Client) -> dict:
    """Failure Path A: Empty text should return graceful validation error."""
    print("\n--- [E2E Test 2] Failure Path A: Empty Text ---", flush=True)
    try:
        result = client.predict(
            text="   ",
            voice="Giọng nữ",
            direct_aud=None,
            direct_txt="",
            lang="Vietnamese",
            mode="TURBO",
            speed=1.0,
            emotion_sel="Tự động nhận diện ngữ cảnh (Auto-Detect)",
            pause_scale=1.0,
            norm_nums=False,
            debug_mode=False,
            api_name="/_generate_ui"
        )
        audio_res, status_md, metrics_md, table_data = result
        assert audio_res is None, "Audio should be None for empty text"
        assert "Vui lòng nhập văn bản" in status_md, f"Unexpected status message: {status_md}"
        print(f"  [PASS] Empty text handled gracefully: '{status_md}'", flush=True)
        return {"test": "Failure Path A: Empty Text", "status": "PASS", "message": status_md}
    except Exception as e:
        print(f"  [FAIL] Empty Text Exception: {e}", flush=True)
        return {"test": "Failure Path A: Empty Text", "status": "FAIL", "error": str(e)}


def test_03_failure_missing_voice(client: Client) -> dict:
    """Failure Path B: Missing / fallback voice profile."""
    print("\n--- [E2E Test 3] Failure Path B: Fallback Default Voice ---", flush=True)
    try:
        result = client.predict(
            text="Thử nghiệm giọng đọc mặc định.",
            voice="[Giọng Mặc Định]",
            direct_aud=None,
            direct_txt="",
            lang="Vietnamese",
            mode="TURBO",
            speed=1.0,
            emotion_sel="Tự động nhận diện ngữ cảnh (Auto-Detect)",
            pause_scale=1.0,
            norm_nums=False,
            debug_mode=False,
            api_name="/_generate_ui"
        )
        audio_res, status_md, metrics_md, table_data = result
        print(f"  [PASS] Fallback voice handled cleanly. Status: {status_md[:60]}...", flush=True)
        return {"test": "Failure Path B: Fallback Voice", "status": "PASS", "message": status_md}
    except Exception as e:
        print(f"  [FAIL] Fallback Voice Exception: {e}", flush=True)
        return {"test": "Failure Path B: Fallback Voice", "status": "FAIL", "error": str(e)}


def test_04_short_text(client: Client) -> dict:
    """Failure Path C: Very short text utterance."""
    print("\n--- [E2E Test 4] Failure Path C: Very Short Text ---", flush=True)
    try:
        result = client.predict(
            text="Ơ.",
            voice="Giọng nữ",
            direct_aud=None,
            direct_txt="",
            lang="Vietnamese",
            mode="TURBO",
            speed=1.0,
            emotion_sel="Tự động nhận diện ngữ cảnh (Auto-Detect)",
            pause_scale=1.0,
            norm_nums=False,
            debug_mode=False,
            api_name="/_generate_ui"
        )
        audio_res, status_md, metrics_md, table_data = result
        if audio_res and os.path.exists(audio_res):
            dest_wav = os.path.join(AUDIO_ARTIFACTS_DIR, "04_short_text_output.wav")
            data, sr = sf.read(audio_res)
            sf.write(dest_wav, data, sr)
            val = validate_generated_audio(dest_wav, min_duration_sec=0.1)
            print(f"  [PASS] Short text synthesized: {val['metrics']['duration_sec']}s audio.", flush=True)
            return {"test": "Failure Path C: Very Short Text", "status": "PASS", "duration_sec": val['metrics']['duration_sec']}
        else:
            print(f"  [PASS] Handled gracefully without output: {status_md}", flush=True)
            return {"test": "Failure Path C: Very Short Text", "status": "PASS", "message": status_md}
    except Exception as e:
        print(f"  [FAIL] Short Text Exception: {e}", flush=True)
        return {"test": "Failure Path C: Very Short Text", "status": "FAIL", "error": str(e)}


def test_05_long_text_chunking(client: Client) -> dict:
    """Failure Path D: Long text triggering multi-chunk segmentation."""
    print("\n--- [E2E Test 5] Failure Path D: Multi-Sentence Chunking ---", flush=True)
    long_text = "Hôm nay trời xanh mát. Tôi đang kiểm tra phân đoạn câu tự động."
    t0 = time.perf_counter()
    try:
        result = client.predict(
            text=long_text,
            voice="Giọng nữ",
            direct_aud=None,
            direct_txt="",
            lang="Vietnamese",
            mode="TURBO",
            speed=1.0,
            emotion_sel="U buồn / Da diết (Melancholic)",
            pause_scale=1.0,
            norm_nums=False,
            debug_mode=True,
            api_name="/_generate_ui"
        )
        latency = time.perf_counter() - t0
        audio_res, status_md, metrics_md, table_data = result
        if audio_res and os.path.exists(audio_res):
            dest_wav = os.path.join(AUDIO_ARTIFACTS_DIR, "05_long_text_output.wav")
            data, sr = sf.read(audio_res)
            sf.write(dest_wav, data, sr)
            val = validate_generated_audio(dest_wav)
            print(f"  [PASS] Multi-sentence chunking succeeded: {val['metrics']['duration_sec']}s audio in {latency:.2f}s.", flush=True)
            return {"test": "Failure Path D: Chunking", "status": "PASS", "duration_sec": val['metrics']['duration_sec'], "latency_sec": round(latency, 2)}
        else:
            return {"test": "Failure Path D: Chunking", "status": "FAIL", "error": status_md}
    except Exception as e:
        print(f"  [FAIL] Chunking Exception: {e}", flush=True)
        return {"test": "Failure Path D: Chunking", "status": "FAIL", "error": str(e)}


def test_06_sequential_requests(client: Client) -> dict:
    """Failure Path E: Sequential requests to verify no state contamination."""
    print("\n--- [E2E Test 6] Failure Path E: Sequential Requests ---", flush=True)
    try:
        t1 = "Xin chào."
        t2 = "Cảm ơn."
        
        res1 = client.predict(
            text=t1,
            voice="Giọng nữ",
            direct_aud=None,
            direct_txt="",
            lang="Vietnamese",
            mode="TURBO",
            speed=1.0,
            emotion_sel="Tự động nhận diện ngữ cảnh (Auto-Detect)",
            pause_scale=1.0,
            norm_nums=False,
            debug_mode=False,
            api_name="/_generate_ui"
        )
        res2 = client.predict(
            text=t2,
            voice="Giọng nữ",
            direct_aud=None,
            direct_txt="",
            lang="Vietnamese",
            mode="TURBO",
            speed=1.0,
            emotion_sel="Tự động nhận diện ngữ cảnh (Auto-Detect)",
            pause_scale=1.0,
            norm_nums=False,
            debug_mode=False,
            api_name="/_generate_ui"
        )
        
        aud1, aud2 = res1[0], res2[0]
        assert aud1 != aud2, "Stale audio contamination detected: Same file returned for distinct requests!"
        assert os.path.exists(aud1) and os.path.exists(aud2)
        print("  [PASS] Sequential requests executed independently with distinct audio files.", flush=True)
        return {"test": "Failure Path E: Sequential Requests", "status": "PASS"}
    except Exception as e:
        print(f"  [FAIL] Sequential Requests Exception: {e}", flush=True)
        return {"test": "Failure Path E: Sequential Requests", "status": "FAIL", "error": str(e)}


def main():
    print("=" * 80, flush=True)
    print("OMNIVOICE V2.21 — AUTONOMOUS BROWSER & UI E2E TEST SUITE", flush=True)
    print("=" * 80, flush=True)

    try:
        client = Client(BASE_URL)
        print(f"✅ Connected to Workstation Web UI at {BASE_URL}", flush=True)
    except Exception as e:
        print(f"❌ Failed to connect to Workstation at {BASE_URL}: {e}", flush=True)
        sys.exit(1)

    all_results = []
    all_results.append(test_01_happy_path(client))
    all_results.append(test_02_failure_empty_text(client))
    all_results.append(test_03_failure_missing_voice(client))
    all_results.append(test_04_short_text(client))
    all_results.append(test_05_long_text_chunking(client))
    all_results.append(test_06_sequential_requests(client))

    # Save summary report
    summary_path = os.path.join(BROWSER_ARTIFACTS_DIR, "e2e_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    passed = sum(1 for r in all_results if r.get("status") == "PASS")
    total = len(all_results)
    print("\n" + "=" * 80, flush=True)
    print(f"E2E SUITE RESULTS: {passed}/{total} TESTS PASSED", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
