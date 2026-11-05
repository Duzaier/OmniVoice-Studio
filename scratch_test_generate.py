"""Quick test to verify generation with Prosody V4 without any ValueError."""
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omnivoice.core.engine import OmniVoiceEngine

print("Initializing engine...")
engine = OmniVoiceEngine.get_instance(device="cpu")
engine.initialize()

test_text = "Có những người xuất hiện đúng lúc trái tim muốn yêu… nhưng lại sai lúc cả hai biết cách ở bên nhau."
print("Synthesizing test text...")
result = engine.synthesize(
    text=test_text,
    language="Vietnamese",
    perf_mode="BALANCED",
    speed=1.0,
    debug_mode=True,
)

print("\nSUCCESS!")
print(f"Audio duration: {result.audio_duration:.2f}s")
print(f"Generation time: {result.generation_time:.2f}s")
print(f"Output path: {result.output_path}")
print(f"Prosody units count: {len(result.prosody_units)}")
