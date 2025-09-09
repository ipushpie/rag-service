import os
import requests
import boto3
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Base URL for your deployed RAGFlow instance
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://148.113.1.127/")

class DocumentInput(BaseModel):
    document_id: str
    source: str  # "postgres" or "minio"

# PostgreSQL configuration
PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'postgres',
    'dbname': 'clm_dev',
}


# MinIO / S3 configuration
MINIO_CONFIG = {
    'endpoint_url': os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    'aws_access_key_id': os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    'aws_secret_access_key': os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    'bucket_name': os.getenv("MINIO_BUCKET", "your-bucket"),
}

def fetch_document(document_id: str, source: str) -> str:
    if source == "postgres":
        conn = psycopg2.connect(**PG_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT content FROM documents WHERE id = %s", (document_id,))
                result = cur.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail="Document not found in PostgreSQL")
                return result[0]
        finally:
            conn.close()
    elif source == "minio":
        s3 = boto3.client('s3',
                          endpoint_url=MINIO_CONFIG['endpoint_url'],
                          aws_access_key_id=MINIO_CONFIG['aws_access_key_id'],
                          aws_secret_access_key=MINIO_CONFIG['aws_secret_access_key'])
        try:
            response = s3.get_object(Bucket=MINIO_CONFIG['bucket_name'], Key=document_id)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not fetch document from MinIO: {e}")
    else:
        raise HTTPException(status_code=400, detail="Invalid source specified (must be 'postgres' or 'minio')")

def upload_to_ragflow(document_text: str, document_id: str):
    files = {'file': (f"{document_id}.txt", document_text)}
    endpoint = f"{RAGFLOW_BASE_URL}/documents"  # Adjust path if different
    resp = requests.post(endpoint, files=files)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Upload failed: {resp.text}")
    return resp.json()

def trigger_chunk_and_ingest(doc_id: str):
    # Example endpoints — adjust based on your actual RAGFlow API paths
    chunk_resp = requests.post(f"{RAGFLOW_BASE_URL}/chunk", json={"document_id": doc_id})
    if not chunk_resp.ok:
        raise HTTPException(status_code=502, detail=f"Chunking failed: {chunk_resp.text}")

    ingest_resp = requests.post(f"{RAGFLOW_BASE_URL}/ingest", json={"document_id": doc_id})
    if not ingest_resp.ok:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {ingest_resp.text}")

@app.post("/process/")
def process_document(input: DocumentInput):
    """
    API endpoint to:
    - Fetch document by ID and source
    - Upload to RAGFlow
    - Trigger chunking & ingestion
    """
    doc_text = fetch_document(input.document_id, input.source)
    upload_result = upload_to_ragflow(doc_text, input.document_id)
    # Assuming RAGFlow returns a JSON with your doc ID
    ragflow_doc_id = upload_result.get("document_id") or input.document_id
    trigger_chunk_and_ingest(ragflow_doc_id)
    return {"status": "success", "document_id": ragflow_doc_id}
