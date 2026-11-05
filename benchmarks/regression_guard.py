import sys
import os
import json
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RegressionGuard")


def check_regression():
    report_file = Path("D:/OmniVoice_Storage/debug/master_performance_report.json")
    if not report_file.exists():
        report_file = Path("./debug/master_performance_report.json")
    
    if not report_file.exists():
        logger.error("No benchmark report found. Run run_master_benchmark.py first.")
        sys.exit(1)

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gen = data["generation"]
    hw = data["hardware"]
    model = data["model"]
    vc = data["voice_clone"]
    qg = data["quality_gates"]

    violations = []

    # Check 1: Macro Chunking (No sentence-level inference)
    if gen["num_generate_calls"] > 10:
        violations.append(f"SENTENCE EXPLOSION: num_generate_calls={gen['num_generate_calls']} exceeds macro chunk threshold.")

    # Check 2: Voice Clone Caching
    if not vc["voice_prompt_cached"]:
        violations.append("PROMPT CACHING FAILED: VoiceClonePrompt is not cached.")

    # Check 3: torch.inference_mode
    if not model["torch_inference_mode"]:
        violations.append("INFERENCE MODE VIOLATION: torch.inference_mode is not enabled.")

    # Check 4: Hard Performance Target (RTF <= 1.33 for 90s)
    if gen["rtf"] > 1.33:
        violations.append(f"HARD PERFORMANCE TARGET EXCEEDED: RTF={gen['rtf']:.3f}x (> 1.33x target for 90s speech).")

    # Check 5: Quality Gates
    if qg["vietnamese_pronunciation"] != "PASS (All 6 tones and consonants verified)":
        violations.append(f"PRONUNCIATION GATE FAILED: {qg['vietnamese_pronunciation']}")

    print("=" * 70)
    print("      🛡️ OMNIVOICE REGRESSION GUARD VERIFICATION")
    print("=" * 70)
    if violations:
        print("❌ REGRESSION DETECTED / PERFORMANCE GATE FAILED:")
        for v in violations:
            print(f"  • {v}")
        print("=" * 70)
        return False
    else:
        print("✅ ALL REGRESSION GUARDS PASSED!")
        print("=" * 70)
        return True


if __name__ == "__main__":
    passed = check_regression()
    sys.exit(0 if passed else 1)
