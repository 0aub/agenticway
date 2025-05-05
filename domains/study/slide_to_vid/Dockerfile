FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Set working dir
WORKDIR /app

# System packages (OCR & PDF conversion)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        poppler-utils \
        tesseract-ocr \
        git \
        build-essential \
        librubberband2 \
        rubberband-cli \
    && rm -rf /var/lib/apt/lists/*

# Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade "optree>=0.13.0"

# Copy source code
COPY src/ ./src
COPY config.yaml ./config.yaml

# Set default command
CMD ["python", "src/main.py"]
