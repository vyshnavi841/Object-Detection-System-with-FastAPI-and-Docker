# Dockerfile

FROM python:3.11-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Configure pip for slow/unstable network connections
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=20

RUN pip install --upgrade pip

# 1. Install PyTorch CPU first (isolated layer)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install OpenCV separately using pre-compiled wheels
RUN pip install --no-cache-dir --prefer-binary opencv-python-headless

# 3. Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and trained weights
COPY src/ ./src/
COPY config.yaml .
COPY best_model.pth .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]