from __future__ import annotations
import shutil
from pathlib import Path
from typing import List

from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from pydub import AudioSegment


def assemble_video(
    slides: List[Path],
    segments: List[AudioSegment],
    out_path: Path,
    frame_rate: int = 24,
) -> Path:
    narration = sum(segments[1:], segments[0])
    tmp_wav = out_path.with_suffix(".tmp.wav")
    narration.export(tmp_wav, format="wav", parameters=["-ar", str(narration.frame_rate)])

    clips = [
        ImageClip(str(img)).set_duration(seg.duration_seconds)
        for img, seg in zip(slides, segments)
    ]
    video = (
        concatenate_videoclips(clips, method="compose")
        .set_audio(AudioFileClip(str(tmp_wav)))
    )
    video.write_videofile(
        str(out_path),
        fps=frame_rate,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        logger=None,
    )

    tmp_wav.unlink(missing_ok=True)
    shutil.rmtree(slides[0].parent, ignore_errors=True)
    return out_path
