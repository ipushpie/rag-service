import os
import requests
import boto3
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Base URL for your deployed RAGFlow instance
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL")
RAGFLOW_DATASET_ID = os.getenv("RAGFLOW_DATASET_ID")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")

class DocumentInput(BaseModel):
    document_id: str
    source: str  # "postgres" or "minio"


# PostgreSQL configuration
PG_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': int(os.getenv('PG_PORT', 5432)),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', 'postgres'),
    'dbname': os.getenv('PG_DBNAME', 'clm_dev'),
}



# MinIO / S3 configuration
MINIO_CONFIG = {
    'endpoint_url': os.getenv("MINIO_ENDPOINT"),
    'aws_access_key_id': os.getenv("MINIO_ACCESS_KEY"),
    'aws_secret_access_key': os.getenv("MINIO_SECRET_KEY"),
    'bucket_name': os.getenv("MINIO_BUCKET"),
}

def fetch_document(document_id: str, source: str):
    if source == "postgres":
        conn = psycopg2.connect(**PG_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT "documentContent", "documentName" FROM "ContractVersion" WHERE "contractId" = %s', (document_id,))
                result = cur.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail="Document not found in PostgreSQL")
                document_content, document_name = result
                return document_content, document_name
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

def upload_to_ragflow(document_text: str, document_name: str):
    files = {'file': (document_name, document_text)}
    endpoint = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{RAGFLOW_DATASET_ID}/documents"
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    resp = requests.post(endpoint, files=files, headers=headers)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Upload failed: {resp.text}")
    return resp.json()

def trigger_chunk_and_ingest(doc_id: str):
    # Use the correct RAGFlow parse documents endpoint
    chunk_endpoint = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{RAGFLOW_DATASET_ID}/chunks"
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"document_ids": [doc_id]}
    chunk_resp = requests.post(chunk_endpoint, json=payload, headers=headers)
    if not chunk_resp.ok:
        raise HTTPException(status_code=502, detail=f"Chunking failed: {chunk_resp.text}")

@app.post("/process/")
def process_document(input: DocumentInput):
    """
    API endpoint to:
    - Fetch document by ID and source
    - Upload to RAGFlow
    - Trigger chunking & ingestion
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    doc_content, doc_name = fetch_document(input.document_id, input.source)
    logging.info(f"Fetched document: {doc_name} for contractId: {input.document_id}")
    upload_result = upload_to_ragflow(doc_content, doc_name)
    logging.info(f"Upload response: {upload_result}")

    # Try to get the document ID from the upload response
    # Extract document ID from upload response (first item in data array)
    ragflow_doc_id = upload_result["data"][0]["id"]
    logging.info(f"Document ID used for parsing: {ragflow_doc_id}")

    # Set chunking method before parsing
    chunk_method = os.getenv("RAGFLOW_CHUNK_METHOD", "naive")  # Change default as needed
    parser_config = {"chunk_token_count": 128}  # Example config, adjust as needed
    update_endpoint = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{RAGFLOW_DATASET_ID}/documents/{ragflow_doc_id}"
    update_payload = {
        "chunk_method": chunk_method,
        "parser_config": parser_config
    }
    update_headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    update_resp = requests.put(update_endpoint, json=update_payload, headers=update_headers)
    logging.info(f"Chunk method update response: {update_resp.text}")

    trigger_chunk_and_ingest(ragflow_doc_id)
    return {"status": "success", "document_id": ragflow_doc_id}
