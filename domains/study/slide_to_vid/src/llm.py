from __future__ import annotations
from functools import lru_cache
from typing import Callable, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# global singletons for reuse
_loaded_gemini = {}
_loaded_local_llm = {}


def build_prompt(slide_text: str, prompt: str, memory: List[str], current_slide_index: int) -> str:
    """
    Generate a calm, teaching-style dialog prompt using [S1] / [S2] tags.
    """
    # Format memory content to indicate previous slides with numbers
    memory_length = len(memory)  # Get the current length of the memory
    memory_content = "\n".join(
        [f"Slide {i + 1}: {mem}" for i, mem in enumerate(memory[-memory_length:])]
    )
    
    # Indicate the current slide
    current_slide_info = f"Current Slide {current_slide_index}: {slide_text}"

    return f"{memory_content}\n\n{current_slide_info}\n\n{prompt}\n\nDialog:"


def _load_local_llm(model_name: str, cache_dir: str) -> Callable[[str], str]:
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    net = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    if torch.cuda.is_available():
        net = net.cuda()

    def _run(prompt: str) -> str:
        out = net.generate(**tok(prompt, return_tensors="pt").to(net.device), max_new_tokens=300)[0]
        return tok.decode(out, skip_special_tokens=True)

    return _run


def _load_gemini(model_name: str, api_key: str) -> Callable[[str], str]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gem = genai.GenerativeModel(model_name)
    return lambda p: gem.generate_content(p).text


def get_llm(provider: str, model_name: str, api_key: str, cache_dir: str) -> Callable[[str], str]:
    global _loaded_gemini, _loaded_local_llm

    if provider == "hf-local":
        if model_name not in _loaded_local_llm:
            _loaded_local_llm[model_name] = _load_local_llm(model_name, cache_dir)
        return _loaded_local_llm[model_name]

    if provider == "google":
        if model_name not in _loaded_gemini:
            _loaded_gemini[model_name] = _load_gemini(model_name, api_key)
        return _loaded_gemini[model_name]

    raise ValueError(f"Unknown provider {provider!r}")
