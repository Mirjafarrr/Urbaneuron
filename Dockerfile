FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    requests==2.32.3 \
    Pillow==10.4.0 \
    albumentations==1.4.15 \
    numpy==1.26.4 \
    torchvision==0.19.1

# Copy source code
COPY src/ ./src/
COPY inference_server.py .
COPY checkpoints/ ./checkpoints/

# Expose the inference port
EXPOSE 8000

# Run the inference server
CMD ["python", "inference_server.py"]