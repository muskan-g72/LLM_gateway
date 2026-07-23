FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 gateway

COPY --chown=gateway:gateway app ./app
COPY --chown=gateway:gateway skills ./skills
COPY --chown=gateway:gateway alembic ./alembic
COPY --chown=gateway:gateway alembic.ini ./

USER gateway
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
