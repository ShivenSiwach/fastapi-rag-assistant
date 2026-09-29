FROM python:3.11-slim

WORKDIR /app

# libgomp1 is a runtime dependency of faiss-cpu on Debian-based images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/processed/ data/processed/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# GEMINI_API_KEY is passed at runtime, not baked into the image:
# docker run -e GEMINI_API_KEY=your-key ...
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]