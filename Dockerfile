# FaceID face-service — production image (CPU).
# GPU uchun: requirements.txt da onnxruntime -> onnxruntime-gpu, bazani nvidia/cuda ga almashtiring.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libglib2.0-0 — opencv-headless, libgomp1 — onnxruntime uchun kerak.
# build-essential faqat insightface (sdist, C-extension) build qilish uchun,
# o'rnatilgach o'sha layerda o'chiriladi.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential g++ \
    && pip install numpy==1.26.4 Cython==3.0.12 \
    && pip install --no-build-isolation insightface==0.7.3 \
    && pip install -r requirements.txt \
    && apt-get purge -y build-essential g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Modellarni build paytida oldindan yuklab qo'yish (alohida cache layer) —
# konteyner starti tezlashadi. Xato bo'lsa build yiqilmaydi: runtime'da
# lazy-load / liveness "disabled" fallback ishlaydi.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])" \
    || echo "WARNING: buffalo_l preload failed; models will be downloaded at runtime"
RUN python -c "import os, urllib.request; os.makedirs('/app/models', exist_ok=True); urllib.request.urlretrieve('https://github.com/hairymax/Face-AntiSpoofing/raw/main/saved_models/AntiSpoofing_bin_1.5_128.onnx', '/app/models/AntiSpoofing_bin_1.5_128.onnx')" \
    || echo "WARNING: liveness model preload failed; will download at runtime or run disabled"

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
