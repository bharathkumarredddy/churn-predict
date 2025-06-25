FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# ✅ Install g++, gcc, and build tools to compile shap
RUN apt-get update && \
    apt-get install -y build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip & install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--workers", "2", "--bind", "0.0.0.0:10000"]
