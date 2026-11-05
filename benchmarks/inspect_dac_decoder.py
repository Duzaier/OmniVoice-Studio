import os
import sys
import inspect
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice

def main():
    print("Loading model...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    tok = model.audio_tokenizer
    print("\n=== audio_tokenizer.decode source ===", flush=True)
    try:
        print(inspect.getsource(tok.decode), flush=True)
    except Exception as e:
        print("Could not get tok.decode source:", e, flush=True)

    print("\n=== acoustic_decoder architecture ===", flush=True)
    print(tok.acoustic_decoder, flush=True)

    print("\n=== acoustic_decoder.forward source ===", flush=True)
    try:
        print(inspect.getsource(type(tok.acoustic_decoder).forward), flush=True)
    except Exception as e:
        print("Could not get source:", e, flush=True)

if __name__ == "__main__":
    main()
