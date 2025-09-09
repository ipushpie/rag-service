# RAG Service

A FastAPI-based backend for document ingestion, chunking, and chat assistant creation using RAGFlow and PostgreSQL/MinIO.

## Features
- Fetch documents from PostgreSQL or MinIO
- Upload documents to RAGFlow
- Trigger chunking and ingestion in RAGFlow
- Create OpenAI-compatible chat assistants
- Environment-based configuration
- Logging for all major operations

## Setup
1. **Clone the repository:**
   ```bash
   git clone git@github-personal:ipushpie/rag-service.git
   cd rag-service
   ```
2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your secrets:
     - PostgreSQL, MinIO, RAGFlow, etc.

## Running the Service
```bash
uvicorn main:app --reload --port 8080
```

## API Endpoints
### 1. Process Document
- **POST** `/process/`
- **Body:**
  ```json
  {
    "document_id": "<doc_id>",
    "source": "postgres"  // or "minio"
  }
  ```
- **Description:** Fetches a document, uploads to RAGFlow, triggers chunking & ingestion.

### 2. Create Chat Assistant
- **POST** `/create_chat_assistant/`
- **Body:**
  ```json
  {
    "name": "my_chat_assistant",
    "dataset_ids": ["<dataset_id>"],
    "prompt": { "prompt": "Your custom prompt here" }
  }
  ```
- **Description:** Creates a chat assistant in RAGFlow. Only `prompt` is required for minimal setup.

## Environment Variables
See `.env.example` for all required variables:
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`
- `RAGFLOW_BASE_URL`, `RAGFLOW_DATASET_ID`, `RAGFLOW_API_KEY`

## Logging
- All major actions are logged using Uvicorn's logger.
- Logs are visible in the console when running with Uvicorn.

## License
MIT

## Author
Pushpender Sharma
