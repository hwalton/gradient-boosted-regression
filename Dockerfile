# Dockerfile.data
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/

ENV PYTHONPATH=/app

CMD ["python", "--help"]