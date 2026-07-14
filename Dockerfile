FROM python:3.10-slim

# Dependências do sistema necessárias para OpenCV headless + Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando para rodar FastAPI no Vercel
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
