FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY aeon_core ./aeon_core
COPY api ./api
COPY worker ./worker

CMD ["python", "-m", "worker.sync"]
