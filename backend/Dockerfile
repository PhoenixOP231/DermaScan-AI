FROM python:3.11-slim

WORKDIR /app

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

# Copy application code + model weights (committed as regular file, not LFS)
COPY . .

# Confirm model file is present and valid
RUN python -c "import os,sys; s=os.path.getsize('model_final.pth'); print('model_final.pth size:',s,'bytes'); sys.exit('ERROR: model file too small') if s<1000000 else print('Model file OK')"

EXPOSE 8080

CMD ["python", "-c", "import os,uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))"]
