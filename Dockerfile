FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Cài Python và các công cụ cần thiết
RUN apt-get update && apt-get install -y python3.10 python3-pip && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip

# Làm việc trong thư mục /app
WORKDIR /app

# Copy code
COPY . /app

# Cài requirements
RUN pip install -r requirements.txt

# Mở port
EXPOSE 8000

# Chạy Redis và FastAPI
CMD sh -c "python Rag/setup_redis.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"
