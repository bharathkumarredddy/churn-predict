FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install system dependencies for SHAP (C++ compiler, build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port (optional, Render handles port via $PORT)
EXPOSE 10000

# Run the Flask app using gunicorn
CMD ["gunicorn", "app:app", "--workers", "2", "--bind", "0.0.0.0:10000"]
