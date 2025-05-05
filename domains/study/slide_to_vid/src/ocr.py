from __future__ import annotations
import tempfile
from pathlib import Path
from typing import List
import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import subprocess

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

def convert_ppt_to_pdf(ppt_path: Path, output_dir: Path) -> Path:
    """Convert .ppt or .pptx to PDF using LibreOffice."""
    pdf_path = output_dir / f"{ppt_path.stem}.pdf"
    
    # Use LibreOffice in headless mode to convert the file
    subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', str(ppt_path), '--outdir', str(output_dir)], check=True)

    return pdf_path

def prepare_file_for_processing(file_path: Path, scratch_dir: Path) -> Path:
    """Check file type and convert to PDF if necessary, then return the PDF path."""
    if file_path.suffix not in ['.pdf']:
        return convert_ppt_to_pdf(file_path, scratch_dir)
    return file_path