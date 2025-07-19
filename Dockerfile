# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all code to container
COPY . /app

# Cài đặt các gói cần thiết
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Cổng FastAPI (uvicorn)
EXPOSE 8000

# Set up redis stack and FastAPI
CMD sh -c "python Rag/setup_redis.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"

