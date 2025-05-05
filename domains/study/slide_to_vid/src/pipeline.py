from __future__ import annotations
import datetime as _dt
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
) -> tts.AudioSegment:
    """OCR → dialog → TTS for one slide, then save .txt and .wav."""
    # ----- generate dialog --------------------------------------------
    slide_text = ocr.ocr_image(img_path)
    prompt      = llm.build_prompt(slide_text)
    dialog      = llm_fn(prompt)

    # ----- prettier logging -------------------------------------------
    preview = " ".join(dialog.split()) # collapse whitespace
    wrapped = textwrap.wrap(preview, width=60) # 60‑char columns
    sample  = "\n".join(wrapped[:2]) # first two lines

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
    audio_seg.export(wav_dir / f"slide_{idx:02}.wav", format="wav")

    return audio_seg


def process_pdf(pdf_path: Path, cfg: AppConfig) -> Path:
    """Full pipeline for one PDF ➜ narrated MP4 (sequential, low‑RAM)."""
    t0 = time.perf_counter()

    # ── run‑specific directories ───────────────────────────────────────
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir   = Path("logs") / f"{cfg.exp_name}_{timestamp}" / pdf_path.stem
    txt_dir, wav_dir = run_dir / "txt", run_dir / "wav"
    for d in (run_dir, txt_dir, wav_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg.save_snapshot(run_dir / "config.yaml")
    init_logger(run_dir)
    log.info(f"📄  Processing {pdf_path.name}")

    # ── slide rasterisation ────────────────────────────────────────────
    slides = ocr.pdf_to_images(pdf_path, cfg.scratch_dir)
    if cfg.test_mode:
        slides = slides[:1]                # only first slide when testing
    total = len(slides)

    # ── LLM loader (lazy singleton) ────────────────────────────────────
    llm_fn = llm.get_llm(
        cfg.provider, cfg.model_name, cfg.google_api_key, str(cfg.models_dir)
    )

    # ── OCR → dialog → TTS for each slide (sequential) ─────────────────
    segments: List[tts.AudioSegment] = []
    for idx, img_path in enumerate(slides, 1):
        seg = _process_slide(
            idx, total, img_path, cfg, txt_dir, wav_dir, llm_fn
        )
        segments.append(seg)

    # ── Assemble video ────────────────────────────────────────────────
    out_video  = run_dir / f"{pdf_path.stem}{cfg.output_suffix}.mp4"
    video_path = video.assemble_video(slides, segments, out_video)

    # ── Extra info: length & size ─────────────────────────────────────
    dur_sec = VideoFileClip(str(video_path)).duration
    size_mb = video_path.stat().st_size / (1024 * 1024)

    dt = _dt.timedelta(seconds=int(time.perf_counter() - t0))
    log.info(
        f"✅  Finished in {dt}. "
        f"Video ➜ {video_path}  |  length {dur_sec:.1f}s  |  {size_mb:.2f} MB"
    )
    return video_path
