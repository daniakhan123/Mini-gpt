FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pydantic numpy pyyaml
COPY src ./src
ENV CHECKPOINT_PATH=checkpoints/tiny_gpt.pt
# Set CHECKPOINT_URL in the hosting dashboard to download the weights at startup.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
