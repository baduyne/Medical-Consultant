# RAG-based Medical Chatbot with VietAI and Redis Stack

This project is a Vietnamese-language chatbot system designed for medical question answering. It uses a **Retrieval-Augmented Generation (RAG)** architecture with the following components:

- **VietAI/vit5-base**: Fine-tuned with LoRA for text generation.
- **Sentence Transformers (all-MiniLM-L6-v2)**: For embedding questions and context.
- **Redis Stack with HNSW (Hierarchical Navigable Small World)**: Vector database for semantic search.
- **FastAPI**: Lightweight backend API.
- **Docker**: Deploy the entire system (Redis Stack + FastAPI backend) in containers using docker-compose.
- **Data Source**: Medical knowledge is derived from public content on [YouMed.vn](https://youmed.vn/).
---

## Features

- Retrieve relevant medical context from a structured knowledge base
- Answer natural language questions using a fine-tuned Vietnamese language model
- Real-time semantic search powered by Redis + HNSW
- Ready-to-deploy with Docker

---

## Architecture Overview
![architecture](/images/medicalchatbot.svg)


## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/baduyne/Medical-Consultant.git
cd Medical-Consultant
```

### 2. Start Redis Stack (with Vector Search)
```bash
docker pull redis/redis-stack:latest
docker run -d --name redis-stack   -p 3107:6379   -p 8001:8001   redis/redis-stack:latest
```

### Build and Run
```bash
docker build -t medical-chatbot .
docker run -p 8000:8000 medical-chatbot
```

---

## Credits
- [VietAI](https://huggingface.co/VietAI/vit5-base) for vit5-base
- [YouMed.vn](https://youmed.vn) for medical content
- HuggingFace, Redis Stack, SentenceTransformers

---
