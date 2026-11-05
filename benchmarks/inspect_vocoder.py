import os
import sys
import inspect
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

def main():
    print("=== OMNIVOICE VOCODER ARCHITECTURE INSPECTION ===", flush=True)
    
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    
    print("Loading model...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    print("\nModel Children:", flush=True)
    for name, child in model.named_children():
        print(f"  • {name}: {type(child)}", flush=True)

    # Inspect audio_tokenizer / vocoder
    tok = model.audio_tokenizer
    print(f"\nAudio Tokenizer / Vocoder Class: {type(tok)}", flush=True)
    print(f"Source file: {inspect.getfile(type(tok))}", flush=True)
    
    # Inspect methods on audio_tokenizer
    methods = [m for m in dir(tok) if not m.startswith("_") and callable(getattr(tok, m))]
    print(f"Public methods: {methods}", flush=True)

    # Look for generator / decoder submodule inside audio_tokenizer
    print("\naudio_tokenizer submodules:", flush=True)
    for name, mod in tok.named_children():
        print(f"  • {name}: {type(mod)}", flush=True)

    # Capture real vocoder input during generation
    print("\nCapturing real vocoder inputs during 1 generation pass...", flush=True)
    vp = vm.load_prompt("default", map_location="cpu")
    text = TextPreprocessor.process("Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo.")

    captured_inputs = []
    
    # Hook the decode / generator method
    orig_decode = tok.decode if hasattr(tok, "decode") else None
    if orig_decode:
        def hook_decode(*args, **kwargs):
            captured_inputs.append({"args": args, "kwargs": kwargs})
            return orig_decode(*args, **kwargs)
        tok.decode = hook_decode

    cfg = OmniVoiceGenerationConfig(num_step=1, guidance_scale=0.0, denoise=False, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate(text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)

    if orig_decode:
        tok.decode = orig_decode

    print(f"\nCaptured {len(captured_inputs)} calls to audio_tokenizer.decode:", flush=True)
    for i, call in enumerate(captured_inputs):
        print(f"\nCall {i+1}:")
        for arg_i, a in enumerate(call["args"]):
            if isinstance(a, torch.Tensor):
                print(f"  Arg {arg_i}: Tensor shape={list(a.shape)}, dtype={a.dtype}, stride={a.stride()}, contiguous={a.is_contiguous()}", flush=True)
            else:
                print(f"  Arg {arg_i}: {type(a)} = {a}", flush=True)
        for k, v in call["kwargs"].items():
            if isinstance(v, torch.Tensor):
                print(f"  Kwarg '{k}': Tensor shape={list(v.shape)}, dtype={v.dtype}, stride={v.stride()}, contiguous={v.is_contiguous()}", flush=True)
            else:
                print(f"  Kwarg '{k}': {type(v)} = {v}", flush=True)

    # Inspect the decoder internal network
    decoder_net = getattr(tok, "generator", getattr(tok, "decoder", None))
    if decoder_net is not None:
        print(f"\nDecoder Network: {type(decoder_net)}", flush=True)
        print(f"Decoder submodules:", flush=True)
        for name, mod in decoder_net.named_children():
            print(f"  • {name}: {type(mod)}", flush=True)

if __name__ == "__main__":
    main()
