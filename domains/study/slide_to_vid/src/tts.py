from __future__ import annotations
import io, os, re, hashlib, random
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal
import torch
import torchaudio
from pydub import AudioSegment

DEFAULT_TURN_GAP = 0.4  # seconds of silence between S1 / S2 lines
DEFAULT_SLIDE_GAP = 1.0  # seconds of silence after each slide


# ───────────────────────────── helpers ────────────────────────────────
def _scaled_gaps(speed: float) -> tuple[int, int]:
    """
    Convert speech_speed (cfg.speech_speed) into pause lengths.

    * speed < 1.0  →  longer pauses   (slower overall pace)
    * speed > 1.0  →  shorter pauses  (faster overall pace)

    Returned values are **milliseconds** ready for AudioSegment.silent().
    """
    factor = 1.0 / speed
    turn_ms = int(DEFAULT_TURN_GAP * factor * 1000)
    slide_ms = int(DEFAULT_SLIDE_GAP * factor * 1000)
    return turn_ms, slide_ms


# ───────────────────────────── Polly (AWS) ────────────────────────────
def _synthesize_polly(
    dialog: str,
    region: str,
    creds: Dict[str, str],
    voices: Dict[str, str],
    speed: float,
) -> AudioSegment:
    """
    Generate Polly audio. Voice rate stays natural (100 %), we only
    stretch/shrink the gaps between lines based on speech_speed.
    """
    import boto3, html

    polly = boto3.client(
        "polly",
        region_name=region,
        aws_access_key_id=creds["access"],
        aws_secret_access_key=creds["secret"],
    )
    TURN_MS, SLIDE_MS = _scaled_gaps(speed)
    sr = 24_000
    out = AudioSegment.silent(0)

    for line in dialog.splitlines():
        m = re.match(r"^\s*\[(S[12])\]\s*(.+)$", line)
        if not m:
            continue
        tag, text = m.groups()
        text = html.escape(text.strip())
        voice_id = voices["s1" if tag == "S1" else "s2"]

        ssml = (
            "<speak>"
            '<amazon:auto-breaths volume="low" frequency="medium" duration="x-short">'
            '<amazon:domain name="conversational">'
            # keep natural voice rate but close with a 100 ms break
            f"<prosody rate='100%'>{text}<break time='100ms'/></prosody>"
            "</amazon:domain>"
            "</amazon:auto-breaths>"
            "</speak>"
        )
        resp = polly.synthesize_speech(
            Text=ssml,
            TextType="ssml",
            VoiceId=voice_id,
            Engine="neural",
            SampleRate=str(sr),
            OutputFormat="mp3",
        )
        out += AudioSegment.from_file(io.BytesIO(resp["AudioStream"].read()), format="mp3")
        SAFETY_MS = 150 # 0.15 s
        out += AudioSegment.silent(SAFETY_MS + TURN_MS)

    return out + AudioSegment.silent(SLIDE_MS)


# ───────────────────────────── Dia (Nari‑Labs) ───────────────────────
@lru_cache(maxsize=1)
def _load_dia(model_name: str, model_cache: Path):
    from dia.model import Dia
    os.environ.setdefault("HF_HOME", str(model_cache))
    return Dia.from_pretrained(model_name)


def _dia_line(
    tag: str,
    text: str,
    model,
    sr: int,
    audio_cache: Path,
    voice_sample: str | None = None,
) -> AudioSegment:
    """
    Generate one [S1]/[S2] line.

    * If `voice_sample` is a valid audio file, Dia clones that voice.
    * Otherwise it falls back to its built-in random speaker.

    The result is cached on disk so each unique (text + sample) combo
    is synthesized only once.
    """
    key_seed = voice_sample or "random"
    key = hashlib.sha1(f"{tag}|{text}|{key_seed}".encode()).hexdigest()
    audio_cache.mkdir(parents=True, exist_ok=True)
    wav_path = audio_cache / f"tts_dia_line_{key}.wav"

    if not wav_path.exists():
        gen_kwargs = {} if voice_sample is None else {"audio_prompt": voice_sample}
        audio_np = model.generate(f"[{tag}] {text}", **gen_kwargs)
        if audio_np is None or audio_np.size == 0:
            raise RuntimeError("Dia returned empty audio")

        torchaudio.save(
            str(wav_path),
            torch.from_numpy(audio_np).unsqueeze(0),
            sample_rate=sr,
            bits_per_sample=16,
        )

    return AudioSegment.from_file(wav_path, format="wav")


def _synthesize_dia(
    dialog: str,
    model_name: str,
    model_cache: Path,
    audio_cache: Path,
    speed: float,
    voices: Dict[str, str], # {"s1": "path/to.wav", "s2": "path2.wav"}
) -> AudioSegment:
    """
    Assemble the full slide audio for Dia:

    * Keeps Dia's native voice quality.
    * Uses voice-sample cloning if paths are provided.
    * Adds silence between lines/slides based on `speech_speed`.
    """
    model = _load_dia(model_name, model_cache)
    sr = 44_100
    TURN_MS, SLIDE_MS = _scaled_gaps(speed)

    s1_sample = voices.get("s1")  # may be None
    s2_sample = voices.get("s2")

    out = AudioSegment.silent(0)

    for line in dialog.splitlines():
        m = re.match(r"^\s*\[(S[12])\]\s*(.+)$", line)
        if not m:
            continue
        tag, text = m.groups()
        sample = s1_sample if tag == "S1" else s2_sample
        seg = _dia_line(tag, text.strip(), model, sr, audio_cache, sample)
        out += seg + AudioSegment.silent(TURN_MS)

    return out + AudioSegment.silent(SLIDE_MS)


# ───────────────────────────── Public API (wrapper) ───────────────────
def synthesize(
    dialog: str,
    engine: Literal["dia", "polly"],
    *,
    model_name: str,
    model_cache: Path,
    audio_cache: Path,
    speed: float,
    region: str,
    creds: Dict[str, str],
    voices: Dict[str, str],
) -> AudioSegment:
    """
    voices:
      * Dia  →  {"s1": "/path/to/voice1.wav", "s2": "/path/voice2.wav"}
      * Polly→  {"s1": "Ivy", "s2": "Brian"}
    """
    if engine == "dia":
        return _synthesize_dia(dialog, model_name, model_cache, audio_cache, speed, voices)
    return _synthesize_polly(dialog, region, creds, voices, speed)
