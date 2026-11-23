FROM python:3.14.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instala primero las dependencias para reutilizar esta capa si cambia el código.
COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir -r requirements-api.txt

# Incluye la API y el modelo versionado con DVC.
COPY MiBici ./MiBici
COPY models/mibici-poisson-v4 ./models/mibici-poisson-v4

# Ejecuta el servicio con un usuario sin privilegios.
RUN addgroup --system --gid 10001 mibici \
    && adduser --system --uid 10001 --ingroup mibici --home /home/mibici mibici
ENV HOME=/home/mibici \
    MIBICI_MODEL_URI=models/mibici-poisson-v4
USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "MiBici.api:app", "--host", "0.0.0.0", "--port", "8000"]
