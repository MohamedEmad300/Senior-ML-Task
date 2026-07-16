"""Shared Ollama helpers.

Discovered the hard way: on this machine (6GB VRAM, RTX 4050 laptop), loading
gemma4:e2b while another model (even a small one like embeddinggemma, ~680MB)
is still resident causes a hard crash in the Ollama runner --
"CUDA error: shared object initialization failed" / stack-buffer-overrun --
NOT a graceful OOM/eviction. This reproduced consistently regardless of
whether requests were concurrent; explicitly stopping other models before
loading a new one (rather than relying on keep_alive eviction) avoided it.
"""
import subprocess


def unload_other_models(keep_model: str):
    """Stop every currently-loaded Ollama model except `keep_model` (call
    before loading a new local model to avoid VRAM-fragmentation crashes)."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=15)
    except Exception as e:
        print(f"WARNING: could not run `ollama ps` to check loaded models: {e}")
        return
    lines = out.stdout.strip().splitlines()[1:]  # skip header
    for line in lines:
        name = line.split()[0] if line.split() else None
        if name and name != keep_model:
            print(f"Unloading Ollama model {name!r} to free VRAM before loading {keep_model!r} ...")
            try:
                subprocess.run(["ollama", "stop", name], capture_output=True, text=True, timeout=30)
            except Exception as e:
                print(f"WARNING: failed to stop {name!r}: {e}")


def unload_all_models():
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=15)
    except Exception as e:
        print(f"WARNING: could not run `ollama ps`: {e}")
        return
    lines = out.stdout.strip().splitlines()[1:]
    for line in lines:
        name = line.split()[0] if line.split() else None
        if name:
            try:
                subprocess.run(["ollama", "stop", name], capture_output=True, text=True, timeout=30)
            except Exception:
                pass
