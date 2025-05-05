from __future__ import annotations
import tempfile
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image
import pytesseract


def pdf_to_images(pdf_path: Path, scratch_dir: Path) -> List[Path]:
    """Split a PDF into PNG slides saved inside *scratch_dir*."""
    tmp_dir = scratch_dir / pdf_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(pdf_path)
    outs: List[Path] = []
    for idx, page in enumerate(pages, 1):
        out = tmp_dir / f"{pdf_path.stem}_s{idx:02}.png"
        page.save(out, "PNG")
        outs.append(out)
    return outs


def ocr_image(img_path: Path) -> str:
    """Return raw OCR text for one slide."""
    return pytesseract.image_to_string(Image.open(img_path))
