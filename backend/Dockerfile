FROM python:3.11-slim

WORKDIR /app

# Install git + git-lfs so Railway can resolve LFS pointers for model_final.pth
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first — ~250 MB vs ~7 GB for CUDA build
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies
RUN pip install --no-cache-dir \
    fastapi==0.135.1 \
    "uvicorn[standard]==0.41.0" \
    python-multipart==0.0.22 \
    opencv-python-headless==4.13.0.92 \
    pillow==12.1.1 \
    numpy==2.4.3 \
    matplotlib==3.10.8

# Copy application code (model_final.pth is tracked via Git LFS)
COPY . .

# If model_final.pth is still an LFS pointer, pull the real file
RUN if [ "$(wc -c < model_final.pth)" -lt 1000 ]; then \
      git lfs pull --include="model_final.pth"; \
    fi

# Confirm model file is present and large enough to be valid
RUN python -c "import os,sys; s=os.path.getsize('model_final.pth'); print('model_final.pth size:',s,'bytes'); sys.exit('ERROR: model_final.pth is an LFS pointer not real file') if s<1000000 else print('Model file OK')"

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
