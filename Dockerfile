# ---------- Stage 1: builder ----------
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: final ----------
FROM python:3.10-slim

RUN addgroup --system appgroup && adduser --system --ingroup appgroup --home /home/appuser --shell /bin/false appuser && mkdir -p /home/appuser && chown appuser:appgroup /home/appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 5000

ENV PORT=5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "--worker-tmp-dir", "/dev/shm", "app.app:app"]