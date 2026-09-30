FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# API port
EXPOSE 1062

# Start FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "1062"]