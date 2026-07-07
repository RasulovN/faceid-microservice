# face-service — FaceID yuz tanish mikroservisi

FaceID SaaS uchun ichki (backend ↔ face-service) Python mikroservis:
yuz aniqlash (RetinaFace), embedding (ArcFace, 512-dim), sifat bahosi,
passiv anti-spoofing (MiniFASNetV2) va pgvector orqali 1:N identifikatsiya.

**Stack:** FastAPI + uvicorn · InsightFace `buffalo_l` · onnxruntime · numpy ·
opencv-python-headless · asyncpg (read-only) · pydantic v2 · structlog (JSON log) · Python 3.11

## Arxitektura

```
face-service/
├── app/
│   ├── main.py            # FastAPI app, lifespan (model load, db pool), endpointlar
│   ├── config.py          # pydantic-settings (env: .env.example bilan mos)
│   ├── security.py        # X-Internal-Api-Key dependency (401 guard)
│   ├── schemas.py         # pydantic v2 request/response modellari
│   ├── logging.py         # structlog JSON konfiguratsiyasi
│   └── services/
│       ├── face.py        # InsightFace wrapper: detect, embedding, quality
│       ├── liveness.py    # MiniFASNet ONNX anti-spoofing (+ "disabled" fallback)
│       └── matcher.py     # cosine similarity, pgvector identify SQL
├── tests/                 # yengil testlar — og'ir ML paketlarsiz o'tadi
├── requirements.txt       # runtime (aniq versiyalar)
├── requirements-dev.txt   # test uchun yengil to'plam
├── Dockerfile             # python:3.11-slim + model preload layer
└── README.md
```

Og'ir importlar (`insightface`, `onnxruntime`, `asyncpg`) **faqat lifespan'da**
amalga oshadi — shu tufayli testlar va modul importi yengil.

## Endpointlar

`GET /health` dan tashqari barchasi `X-Internal-Api-Key: <INTERNAL_API_KEY>`
headerini talab qiladi, aks holda **401**.

| Endpoint | Tavsif |
|---|---|
| `POST /extract` | multipart `images` (1..5). Har rasm: `{ok, embedding: [512] \| null, quality, error}`. Xatolar: `FACE_NOT_FOUND`, `FACE_MULTIPLE`, `FACE_LOW_QUALITY` (quality < 0.35), `INVALID_IMAGE`. |
| `POST /verify` | JSON `{image_b64, embeddings: [[512]...], match_threshold?, check_liveness}` → `{match, similarity, liveness_score, liveness_passed, error}`. Threshold default: `FACE_MATCH_THRESHOLD`. |
| `POST /identify` | JSON `{image_b64, company_id (uuid), check_liveness}` → pgvector top-5 qidiruv → `{found, employee_id, similarity, liveness_score, liveness_passed, error}`. |
| `POST /liveness` | JSON `{image_b64}` → `{liveness_score, passed, error}`. |
| `GET /health` | `{status: "ok"\|"degraded", model_loaded, db: "ok"\|"error", liveness: "ok"\|"disabled"}` — api-key talab qilinmaydi. |

Semantika bo'yicha muhim eslatmalar:

- **extract** (enrollment) qat'iy: rasmda **aynan bitta** yuz bo'lishi shart.
- **verify / identify / liveness** kiosk-rejimga mos: bir nechta yuz bo'lsa
  **eng katta** yuz olinadi (orqadagi odamlar xalaqit bermasligi uchun).
- `match`/`found` va `liveness_passed` **mustaqil** qaytariladi — yakuniy qaror
  (masalan, `LIVENESS_FAILED` xatosi) backend tomonida qabul qilinadi.
- Liveness inference'da runtime xato bo'lsa **fail-closed**: `score=0.0`
  (log: `liveness_inference_failed`).

## Sifat bahosi (quality, 0..1)

`quality = 0.5·det_score + 0.25·size + 0.25·sharpness`

- `det_score` — RetinaFace ishonch bahosi;
- `size` — yuzning kichik tomoni / 112px (ArcFace kirish o'lchami), 1.0 da to'yinadi;
- `sharpness` — 112×112 grayscale cropdagi Laplacian variance / 100, 1.0 da to'yinadi.

`quality < FACE_QUALITY_THRESHOLD` (default **0.35**) → `FACE_LOW_QUALITY`.

## Liveness yechimi

**Asosiy yo'l:** Silent-Face-Anti-Spoofing loyihasidagi **MiniFASNetV2** modelining
tayyor ONNX konversiyasi ([hairymax/Face-AntiSpoofing](https://github.com/hairymax/Face-AntiSpoofing),
`AntiSpoofing_bin_1.5_128.onnx` — binary live/spoof, 128×128 kirish, bbox 1.5×
kengaytiriladi, live class index = 0). Model birinchi ishga tushishda
`LIVENESS_MODEL_URL` dan yuklab olinadi (Dockerfile build paytida preload qiladi).

Preprocessing: yuz bbox'i kvadrat qilib 1.5× kengaytiriladi (chetlar 0 bilan
to'ldiriladi) → RGB → 128×128 → CHW float32/255 → softmax → `probs[0]` = live ehtimoli.

**Fallback ("disabled" rejim):** model yuklab olinmasa/ochilmasa servis startda
`liveness_disabled` WARNING loglaydi va barcha so'rovlarga
`liveness_score=1.0, passed=true` qaytaradi; `/health` da `liveness: "disabled"`
ko'rinadi. Heuristik pseudo-liveness ataylab yozilmagan.

Boshqa MiniFASNet ONNX modelini ulash uchun: `LIVENESS_MODEL_URL`,
`LIVENESS_MODEL_PATH`, `LIVENESS_INPUT_SIZE`, `LIVENESS_BBOX_SCALE`,
`LIVENESS_LIVE_INDEX` env'larini sozlang (masalan, original 3-klassli
`2.7_80x80_MiniFASNetV2` konversiyasida live index = 1, input 80).

## DB konvensiyasi (identify)

Backend TypeORM default'ida ustunlar **camelCase quoted**, jadvallar esa
**snake_case** deb qabul qilingan (shu servis va backend migratsiyalari uchun
YAGONA kelishuv):

- jadvallar: `face_embeddings`, `employees`
- ustunlar: `"employeeId"`, `"companyId"`, `"deletedAt"`, `status`, `id`, `embedding`

SQL (`app/services/matcher.py` → `IDENTIFY_SQL`):

```sql
SELECT fe."employeeId", 1 - (fe.embedding <=> $1::vector) AS similarity
FROM face_embeddings fe
JOIN employees e ON e.id = fe."employeeId"
WHERE e."companyId" = $2 AND e.status != 'FIRED' AND e."deletedAt" IS NULL
ORDER BY fe.embedding <=> $1::vector LIMIT 5
```

Agar backend sxemasi boshqacha nomlansa — faqat `IDENTIFY_SQL` konstantasini
o'zgartirish kifoya. Ulanish `FACE_SERVICE_DATABASE_URL` orqali, pool
`default_transaction_read_only=on` bilan ochiladi (servis DB'ga yozmaydi).

## Env o'zgaruvchilar

Majburiy/asosiylari `.env.example` (repo ildizida) bilan bir xil:

| Env | Default | Tavsif |
|---|---|---|
| `INTERNAL_API_KEY` | — (bo'sh = hamma so'rov 401) | Node ↔ Python ichki kalit |
| `FACE_MATCH_THRESHOLD` | `0.5` | 1:1 / 1:N cosine similarity chegarasi |
| `LIVENESS_THRESHOLD` | `0.7` | Anti-spoofing chegarasi |
| `FACE_SERVICE_DATABASE_URL` | — (bo'sh = /identify 503) | Read-only pgvector ulanish |
| `FACE_USE_GPU` | `false` | `true` → CUDAExecutionProvider (onnxruntime-gpu kerak) |
| `FACE_SERVICE_PORT` | `8000` | `python -m app.main` uchun port |
| `FACE_QUALITY_THRESHOLD` | `0.35` | Enrollment sifat chegarasi |
| `FACE_MODEL_NAME` / `FACE_DET_SIZE` | `buffalo_l` / `640` | InsightFace pack va det o'lchami |
| `LIVENESS_MODEL_URL` / `LIVENESS_MODEL_PATH` | hairymax ONNX / `models/...onnx` | Liveness model manbai |
| `LIVENESS_INPUT_SIZE` / `LIVENESS_BBOX_SCALE` / `LIVENESS_LIVE_INDEX` | `128` / `1.5` / `0` | Model preprocessing parametrlari |
| `LOG_LEVEL` | `INFO` | structlog/stdlib daraja |

## Ishga tushirish

```bash
# To'liq (og'ir) muhit
python -m venv .venv && .venv/Scripts/activate       # Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker build -t faceid/face-service .
docker run --rm -p 8000:8000 \
  -e INTERNAL_API_KEY=... \
  -e FACE_SERVICE_DATABASE_URL=postgresql://faceid:...@postgres:5432/faceid \
  faceid/face-service
```

Dockerfile build paytida `buffalo_l` va liveness ONNX modellarini alohida
layerga preload qiladi (start tez); preload muvaffaqiyatsiz bo'lsa build
yiqilmaydi — modellar runtime'da lazy yuklanadi.

## Testlar

Testlar og'ir ML stack'siz o'tadi (modellar faqat lifespan'da yuklanadi,
testlar lifespan'ga kirmaydi — endpointlar `app.state`ga qo'yilgan fake'lar
bilan tekshiriladi):

```bash
pip install -r requirements-dev.txt
pytest
```

Qamrov: quality funksiyasi, blur/decode helperlar, cosine/pgvector matcher,
liveness preprocessing + disabled fallback, api-key guard (401), pydantic
schema validatsiyasi (422), extract/verify/identify/liveness/health oqimlari.

## GPU

`FACE_USE_GPU=true` + `onnxruntime` o'rniga `onnxruntime-gpu==1.21.1`
o'rnating (CUDA bazali image kerak). Provider ro'yxati avtomatik
`["CUDAExecutionProvider", "CPUExecutionProvider"]` bo'ladi.
# faceid-microservice
