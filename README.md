# RAG-based Medical Chatbot with T5 and Redis Stack

This project is a Vietnamese-language chatbot system designed for medical question answering. It uses a **Retrieval-Augmented Generation (RAG)** architecture with the following components:

- **VietAI/vit5-base**: Fine-tuned with LoRA for text generation.
- **Sentence Transformers (vinai/phobert-base)**: For embedding titles, questions, context.
- **Redis Stack with HNSW (Hierarchical Navigable Small World)**: Vector database for semantic search.
- **FastAPI**: Lightweight backend API.
- **Docker**: Deploy the entire system (Redis Stack + FastAPI backend) in containers using docker-compose.
- **Data Source**: Medical knowledge is derived from public content on [YouMed.vn](https://youmed.vn/).
    - Note:
        - The raw crawled dataset is not provided due to data protection policies.
        - The data collection and preprocessing pipeline is not included in this repository.
- **Trained Model**: [baduyne/vnt5-medical-gqa](https://huggingface.co/baduyne/vnt5-medical-gqa/tree/main)
- **Evaluate Metric**: BLEU = 41.4
---

## Features

- Retrieve relevant medical context from a structured knowledge base
- Answer natural language questions using a fine-tuned Vietnamese language model
- Real-time semantic search powered by Redis + HNSW
- Ready-to-deploy with Docker

---

## Architecture Overview
<img src="/images/medicalchatbot.svg" alt="architecture" width="600"/>


## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/baduyne/Medical-Consultant.git
cd Medical-Consultant
```

### 2. Redis Stack + App
```bash
docker-compose up --build
docker run --gpus all -p 8000:8000 your_image_name
```
```bash
# Set up nvidia-container-toolkit:
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
---

## Credits
- [VietAI](https://huggingface.co/VietAI/vit5-base) for vit5-base
- [YouMed.vn](https://youmed.vn) for medical content
- HuggingFace, Redis Stack, SentenceTransformers

---


