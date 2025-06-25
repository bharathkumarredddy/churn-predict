FROM python:3.9.18-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip==20.3.4 \
    && pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Create static folder if it doesn't exist
RUN mkdir -p static

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "app:app"]