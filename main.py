import os
import requests
import boto3
import psycopg2
import asyncio
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)


app = FastAPI()

# Load environment variables
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
    'host': os.getenv('POSTGRES_HOST'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'dbname': os.getenv('POSTGRES_DB'),
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
        try:
            conn = psycopg2.connect(**PG_CONFIG)
            cursor = conn.cursor()
            cursor.execute('SELECT "documentContent", "documentName" FROM "ContractVersion" WHERE "contractId" = %s', (document_id,))
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Document not found in PostgreSQL")
            document_content, document_name = result
            return document_content, document_name
        finally:
            cursor.close()
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

def check_document_progress(document_id: str):
    """
    Check the parsing progress of a document using the chunks API.
    Returns the document progress information.
    """
    endpoint = f"{RAGFLOW_BASE_URL}/api/v1/datasets/{RAGFLOW_DATASET_ID}/documents/{document_id}/chunks"
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    params = {
        "page": 1,
        "page_size": 1  # We only need the doc info, not the actual chunks
    }
    
    resp = requests.get(endpoint, headers=headers, params=params)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Progress check failed: {resp.text}")
    
    data = resp.json()
    if data.get("code") != 0:
        raise HTTPException(status_code=502, detail=f"API error: {data}")
    
    return data["data"]["doc"]

async def monitor_document_progress(document_id: str, max_wait_time: int = 300, poll_interval: int = 5):
    """
    Monitor document parsing progress by polling the API regularly.
    
    Args:
        document_id: The Ragflow document ID to monitor
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
        poll_interval: How often to check progress in seconds (default: 5 seconds)
    
    Returns:
        dict: Final document status with progress information
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            doc_info = check_document_progress(document_id)
            progress = doc_info.get("progress", 0)
            status = doc_info.get("status", "unknown")
            progress_msg = doc_info.get("progress_msg", "")
            
            logging.info(f"Document {document_id} - Progress: {progress}%, Status: {status}, Message: {progress_msg}")
            
            # Check if parsing is complete
            # Status "1" typically means completed, progress 1.0 means 100%
            if status == "1" and progress >= 1.0:
                logging.info(f"Document {document_id} parsing completed successfully!")
                return {
                    "status": "completed",
                    "document_info": doc_info,
                    "total_wait_time": time.time() - start_time
                }
            
            # Check for error states
            if status in ["-1", "2"]:  # Common error status codes
                logging.error(f"Document {document_id} parsing failed with status {status}")
                return {
                    "status": "failed",
                    "document_info": doc_info,
                    "total_wait_time": time.time() - start_time
                }
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
            
        except Exception as e:
            logging.error(f"Error checking progress for document {document_id}: {e}")
            await asyncio.sleep(poll_interval)
    
    # Timeout reached
    logging.warning(f"Timeout reached while monitoring document {document_id}")
    try:
        final_doc_info = check_document_progress(document_id)
        return {
            "status": "timeout",
            "document_info": final_doc_info,
            "total_wait_time": max_wait_time
        }
    except:
        return {
            "status": "timeout",
            "document_info": None,
            "total_wait_time": max_wait_time
        }

@app.post("/process/")
def process_document(input: DocumentInput):
    """
    API endpoint to:
    - Fetch document by ID and source
    - Upload to RAGFlow
    - Trigger chunking & ingestion
    """


    doc_content, doc_name = fetch_document(input.document_id, input.source)
    logging.info(f"Fetched document: {doc_name} for contractId: {input.document_id}")
    upload_result = upload_to_ragflow(doc_content, doc_name)
    logging.info(f"Upload response: {upload_result}")

    # Try to get the document ID from the upload response
    # Extract document ID from upload response (first item in data array)
    ragflow_doc_id = upload_result["data"][0]["id"]
    logging.info(f"Document ID used for parsing: {ragflow_doc_id}")


    trigger_chunk_and_ingest(ragflow_doc_id)
    return {"status": "success", "document_id": ragflow_doc_id}

@app.post("/process_with_monitoring/")
async def process_document_with_monitoring(input: DocumentInput):
    """
    API endpoint to:
    - Fetch document by ID and source
    - Upload to RAGFlow
    - Trigger chunking & ingestion
    - Monitor parsing progress until completion
    - Automatically create chat assistant for summarization
    """
    try:
        # Fetch document
        doc_content, doc_name = fetch_document(input.document_id, input.source)
        logging.info(f"Fetched document: {doc_name} for contractId: {input.document_id}")
        
        # Upload to RAGFlow
        upload_result = upload_to_ragflow(doc_content, doc_name)
        logging.info(f"Upload response: {upload_result}")

        # Extract document ID from upload response
        ragflow_doc_id = upload_result["data"][0]["id"]
        logging.info(f"Document ID used for parsing: {ragflow_doc_id}")

        # Trigger chunking & ingestion
        trigger_chunk_and_ingest(ragflow_doc_id)
        
        # Monitor progress
        logging.info(f"Starting progress monitoring for document {ragflow_doc_id}")
        progress_result = await monitor_document_progress(ragflow_doc_id)
        
        # Create chat assistant automatically after successful parsing
        chat_assistant_result = None
        session_result = None
        summary_result = None
        
        if progress_result.get("status") == "completed":
            logging.info(f"Document parsing completed, creating chat assistant for document {input.document_id}")
            
            # Create chat assistant with simplified approach (no complex prompt)
            assistant_name = f"Summary Assistant - {doc_name}"
            chat_assistant_result = create_chat_assistant(
                name=assistant_name,
                dataset_ids=[RAGFLOW_DATASET_ID],
                avatar=""
            )
            logging.info(f"Chat assistant created: {chat_assistant_result}")
            
            # If assistant was created successfully, create a session and get summary
            if chat_assistant_result and chat_assistant_result.get("code") == 0:
                chat_id = chat_assistant_result["data"]["id"]
                logging.info(f"Creating session for chat assistant {chat_id}")
                
                # Create session
                session_result = create_chat_session(
                    chat_id=chat_id,
                    session_name=f"Summary Session - {doc_name}"
                )
                logging.info(f"Session created: {session_result}")
                
                # If session was created successfully, get document summary
                if session_result and session_result.get("code") == 0:
                    session_id = session_result["data"]["id"]
                    logging.info(f"Getting document summary for session {session_id}")
                    
                    # Get summary
                    summary_result = get_document_summary(
                        chat_id=chat_id,
                        session_id=session_id,
                        document_name=doc_name
                    )
                    logging.info(f"Summary generated: {summary_result}")
                else:
                    logging.error(f"Failed to create session: {session_result}")
            else:
                logging.error(f"Failed to create chat assistant: {chat_assistant_result}")
                # Try to continue anyway - maybe assistant was partially created
                if chat_assistant_result and "data" in chat_assistant_result and "id" in chat_assistant_result["data"]:
                    chat_id = chat_assistant_result["data"]["id"]
                    logging.info(f"Attempting to create session with partially created assistant {chat_id}")
                    
                    session_result = create_chat_session(
                        chat_id=chat_id,
                        session_name=f"Summary Session - {doc_name}"
                    )
                    logging.info(f"Session created: {session_result}")
                    
                    if session_result and session_result.get("code") == 0:
                        session_id = session_result["data"]["id"]
                        summary_result = get_document_summary(
                            chat_id=chat_id,
                            session_id=session_id,
                            document_name=doc_name
                        )
                        logging.info(f"Summary generated: {summary_result}")
        
        return {
            "status": "success",
            "document_id": ragflow_doc_id,
            "original_document_id": input.document_id,
            "source": input.source,
            "monitoring_result": progress_result,
            "chat_assistant": chat_assistant_result,
            "session": session_result,
            "document_summary": summary_result
        }
        
    except Exception as e:
        logging.error(f"Error processing document {input.document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_progress/{document_id}")
async def check_progress(document_id: str):
    """
    API endpoint to check the current parsing progress of a document.
    """
    try:
        doc_info = check_document_progress(document_id)
        return {
            "status": "success",
            "document_id": document_id,
            "progress": doc_info.get("progress", 0),
            "status_code": doc_info.get("status", "unknown"),
            "progress_message": doc_info.get("progress_msg", ""),
            "document_info": doc_info
        }
    except Exception as e:
        logging.error(f"Error checking progress for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/create_chat_assistant/")
async def api_create_chat_assistant(request: Request):
    body = await request.json()
    name = body.get("name")
    dataset_ids = body.get("dataset_ids")
    avatar = body.get("avatar", "")
    llm = body.get("llm")
    prompt = body.get("prompt")
    result = create_chat_assistant(
        name=name,
        dataset_ids=dataset_ids,
        avatar=avatar,
        llm=llm,
        prompt=prompt
    )
    return JSONResponse(content=result)

@app.post("/create_session_and_summary/{chat_id}")
async def create_session_and_get_summary(chat_id: str, request: Request):
    """
    Create a session for an existing chat assistant and get document summary.
    """
    try:
        body = await request.json()
        session_name = body.get("session_name", "Document Summary Session")
        document_name = body.get("document_name", "uploaded document")
        
        # Create session
        session_result = create_chat_session(chat_id=chat_id, session_name=session_name)
        
        if session_result and session_result.get("code") == 0:
            session_id = session_result["data"]["id"]
            
            # Get summary
            summary_result = get_document_summary(
                chat_id=chat_id,
                session_id=session_id,
                document_name=document_name
            )
            
            return {
                "status": "success",
                "chat_id": chat_id,
                "session": session_result,
                "summary": summary_result
            }
        else:
            raise HTTPException(status_code=502, detail=f"Failed to create session: {session_result}")
            
    except Exception as e:
        logging.error(f"Error creating session and summary for chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_chat_assistant(
    name,
    dataset_ids,
    avatar="",
    llm=None,
    prompt=None
):
    """
    Create a chat assistant in RAGFlow.
    Args:
        name (str): Name of the chat assistant.
        dataset_ids (list): List of dataset IDs.
        avatar (str): Base64 avatar string (optional).
        llm (dict): LLM config (optional).
        prompt (dict): Prompt config (optional).
    Returns:
        dict: Response from RAGFlow API.
    """
    RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL")
    RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
    url = f"{RAGFLOW_BASE_URL}/api/v1/chats"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    
    # Use simple payload structure as shown in your curl example
    payload = {
        "dataset_ids": dataset_ids,
        "name": name
    }
    
    # Only add optional fields if provided
    if avatar:
        payload["avatar"] = avatar
    if llm:
        payload["llm"] = llm
    if prompt:
        payload["prompt"] = prompt
        
    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()
    except Exception as e:
        logging.error(f"Error creating chat assistant: {e}")
        return {"error": response.text}

def create_chat_session(chat_id: str, session_name: str = "Document Summary Session"):
    """
    Create a chat session with the assistant.
    Args:
        chat_id (str): The chat assistant ID.
        session_name (str): Name for the session.
    Returns:
        dict: Response from RAGFlow API.
    """
    RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL")
    RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
    url = f"{RAGFLOW_BASE_URL}/api/v1/chats/{chat_id}/sessions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    payload = {
        "name": session_name
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()
    except Exception:
        return {"error": response.text}

def get_document_summary(chat_id: str, session_id: str, document_name: str):
    """
    Send a message to get document summary from the chat assistant using completions API.
    Args:
        chat_id (str): The chat assistant ID.
        session_id (str): The session ID.
        document_name (str): Name of the document for context.
    Returns:
        dict: Response from RAGFlow API.
    """
    RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL")
    RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
    url = f"{RAGFLOW_BASE_URL}/api/v1/chats/{chat_id}/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }
    
    # Request a comprehensive summary
    summary_request = f"Please provide a comprehensive summary of the document '{document_name}'. Include the main topics, key points, important details, and any significant findings or conclusions. Structure the summary in a clear and organized manner."
    
    payload = {
        "question": summary_request,
        "stream": False,  # Set to False to get complete response at once
        "session_id": session_id
    }
    
    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()
    except Exception as e:
        logging.error(f"Error getting document summary: {e}")
        return {"error": response.text}

