from __future__ import annotations
import datetime
import logging
import textwrap
import time
from pathlib import Path
from typing import List, Callable
from moviepy.editor import VideoFileClip

from config import AppConfig
from logger import init_logger
import ocr, llm, tts, video

log = logging.getLogger("slide_to_vid")


def _process_slide(
    idx: int,
    total: int,
    img_path: Path,
    cfg: AppConfig,
    txt_dir: Path,
    wav_dir: Path,
    llm_fn: Callable[[str], str],
    memory: List[str]
) -> tts.AudioSegment:
    """OCR → dialog → TTS for one slide, then save .txt and .wav."""
    start_time = time.time()  # Start timing

    # ----- generate dialog --------------------------------------------
    slide_text = ocr.ocr_image(img_path)
    prompt = llm.build_prompt(slide_text, cfg.prompt, memory, idx)
    dialog = llm_fn(prompt)

    # ----- prettier logging -------------------------------------------
    preview = " ".join(dialog.split()) # collapse whitespace
    wrapped = textwrap.wrap(preview, width=60) # 60‑char columns
    sample = "\n".join(wrapped[:2]) # first two lines

    log.info(f"[{idx}/{total}] {sample}")

    # ----- TTS ---------------------------------------------------------
    audio_seg = tts.synthesize(
        dialog,
        cfg.tts_engine.lower(),
        model_name=cfg.tts_model,
        model_cache=cfg.models_dir,
        audio_cache=cfg.scratch_dir,
        speed=cfg.speech_speed,
        region=cfg.aws_region,
        creds={"access": cfg.aws_access_key, "secret": cfg.aws_secret_key},
        voices=(
            cfg.dia_voices
            if cfg.tts_engine.lower() == "dia"
            else {k: getattr(v, 'id', v) for k, v in cfg.polly_voices.items()}
        ),
    )

    # ── persist artifacts ───────────────────────────────────────────────
    (txt_dir / f"slide_{idx:02}.txt").write_text(dialog)
    audio_file_path = wav_dir / f"slide_{idx:02}.wav"
    audio_seg.export(audio_file_path, format="wav")

    # Store the generated dialog in memory
    memory.append(dialog)
    if len(memory) > cfg.memory_length:
        memory.pop(0)  # Remove the oldest entry if memory exceeds the length

    # ----- Log processing time and audio length -----------------------
    processing_time = time.time() - start_time
    audio_length = audio_seg.duration_seconds  # Get audio length in seconds
    audio_length_hh_mm_ss = str(datetime.timedelta(seconds=int(audio_length)))
    processing_time_hh_mm_ss = str(datetime.timedelta(seconds=int(processing_time)))

    log.info(f" -->  Processing time: {processing_time_hh_mm_ss}\n -->  Audio length: {audio_length_hh_mm_ss}")

    return audio_seg


def process_pdf(file_path: Path, cfg: AppConfig, exp_dir: Path) -> Path:
    """Full pipeline for one PDF or PPT ➜ narrated MP4 (sequential, low‑RAM)."""
    start = time.time()

    # ── run‑specific directories ───────────────────────────────────────
    run_dir = exp_dir / file_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for txt and wav
    txt_dir = run_dir / "txt"
    wav_dir = run_dir / "wav"
    txt_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    init_logger(run_dir)
    log.info(f"📄  Processing {file_path.name}")

    # ── Prepare file for processing ───────────────────────────────────
    file_path = ocr.prepare_file_for_processing(file_path, cfg.scratch_dir)

    # ── slide rasterisation ────────────────────────────────────────────
    slides = ocr.pdf_to_images(file_path, cfg.scratch_dir)
    if cfg.test_mode:
        slides = slides[:1]  # only first slide when testing
    total = len(slides)

    # ── LLM loader (lazy singleton) ────────────────────────────────────
    llm_fn = llm.get_llm(
        cfg.provider, cfg.model_name, cfg.google_api_key, str(cfg.models_dir)
    )

    # ── OCR → dialog → TTS for each slide (sequential) ─────────────────
    segments: List[tts.AudioSegment] = []
    memory = []
    for idx, img_path in enumerate(slides, 1):
        seg = _process_slide(
            idx, total, img_path, cfg, txt_dir, wav_dir, llm_fn, memory
        )
        segments.append(seg)

    # ── Assemble video ────────────────────────────────────────────────
    out_video  = run_dir / f"{file_path.stem}{cfg.output_suffix}.mp4"
    video_path = video.assemble_video(slides, segments, out_video)

    # ── Extra info: length & size ─────────────────────────────────────
    dur_sec = VideoFileClip(str(video_path)).duration
    size_mb = video_path.stat().st_size / (1024 * 1024)

    # Format duration as hh:mm:ss
    elapsed_time = datetime.timedelta(seconds=int(time.time() - start)) 
    dur_hh_mm_ss = datetime.timedelta(seconds=int(dur_sec))
    log.info(
        f"✅  Finished in {elapsed_time}. "
        f"Video ➜ {video_path}  |  length {dur_hh_mm_ss}  |  {size_mb:.2f} MB"
    )
    return video_path
