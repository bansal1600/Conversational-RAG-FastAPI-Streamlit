import os
import sys
import uuid
import logging
import shutil   

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging FIRST before any other imports
from config.settings import LOGGING_CONFIG

logging.basicConfig(
    filename=str(LOGGING_CONFIG["log_file"]), 
    level=getattr(logging, LOGGING_CONFIG["log_level"]),
    format=LOGGING_CONFIG["log_format"],
    filemode=LOGGING_CONFIG["file_mode"]
)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from src.models.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from src.core.langchain_utils import get_rag_chain
from src.database.db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from src.database.chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from src.cache.redis_utils import redis_cache
from src.database.connection_manager import connection_manager
from src.cache.semantic_cache import semantic_cache, context_optimizer
from config.settings import API_CONFIG

app = FastAPI(title="Conversational RAG API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to Conversational RAG API!", "version": "1.0.0"}

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Step 1: Check semantic cache for similar queries
    cached_response = semantic_cache.find_similar_cached_response(
        query_input.question, 
        query_input.api_key
    )
    
    if cached_response:
        logging.info(f"Semantic cache HIT: {cached_response['similarity']:.3f} similarity")
        answer = cached_response['response']
        # Still log the interaction but mark it as cached
        insert_application_logs(session_id, query_input.question, f"[CACHED] {answer}", query_input.model.value)
        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)
    
    # Step 2: Get and optimize chat history
    raw_chat_history = get_chat_history(session_id)
    optimized_history, context_summary = context_optimizer.optimize_context(raw_chat_history, query_input.api_key)
    
    if context_summary:
        logging.info(f"Context optimized: {len(raw_chat_history)} -> {len(optimized_history)} messages + summary")
        # Prepend summary to optimized history
        optimized_history.insert(0, {"role": "system", "content": f"Previous context summary: {context_summary}"})
    
    logging.info(f"fetched user history successful")
    
    # Step 3: Get RAG chain and process query
    rag_chain = get_rag_chain(query_input.api_key, query_input.model.value)
    logging.info(f"building rag chain complete")
    
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": optimized_history
    })['answer']
    logging.info(f"llm response")
    
    # Step 4: Cache the response semantically
    semantic_cache.cache_response(
        query_input.question,
        answer,
        query_input.api_key,
        ttl_hours=6
    )
    
    # Step 5: Store in database
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...), api_key: str = Form(...), session_id: str = Form(...)):
    allowed_extensions = API_CONFIG["allowed_file_extensions"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"at before insert_document_record")
        try:
            file_id = insert_document_record(file.filename, session_id)
            logging.info(f"Document record inserted with file_id: {file_id}")
        except Exception as e:
            logging.error(f"Error inserting document record: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        logging.info(f"at before index_document_to_chroma")
        try:
            success = index_document_to_chroma(temp_file_path, file_id, api_key)
            
            if success:
                logging.info(f"Successfully indexed document {file.filename} with file_id: {file_id}")
                return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
            else:
                logging.error(f"Failed to index document {file.filename}")
                delete_document_record(file_id)
                raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
        except Exception as e:
            logging.error(f"Exception during document indexing: {e}")
            delete_document_record(file_id)
            # Return more specific error message
            if "API key" in str(e).lower():
                raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
            else:
                raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents(session_id: str = None):
    return get_all_documents(session_id)

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id, request.api_key)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}

@app.get("/cache-stats")
def get_cache_stats():
    """Get Redis cache statistics and performance metrics"""
    return {
        "redis_stats": redis_cache.get_cache_stats(),
        "cache_status": "enabled" if redis_cache.is_connected() else "disabled"
    }

@app.get("/connection-stats")
def get_connection_stats():
    """Get database and vector store connection statistics"""
    return {
        "connection_stats": connection_manager.get_connection_stats(),
        "status": "optimized"
    }

@app.get("/semantic-cache-stats")
def get_semantic_cache_stats():
    """Get semantic cache statistics and performance metrics"""
    if not redis_cache.is_connected():
        return {"error": "Redis not connected"}
    
    try:
        # Count semantic cache entries
        semantic_keys = redis_cache.redis_client.keys("semantic_cache:*")
        embedding_keys = redis_cache.redis_client.keys("embedding:*")
        
        return {
            "semantic_cache_entries": len(semantic_keys),
            "cached_embeddings": len(embedding_keys),
            "similarity_threshold": semantic_cache.similarity_threshold,
            "max_history_length": semantic_cache.max_history_length,
            "context_optimizer": {
                "max_context_tokens": context_optimizer.max_context_tokens,
                "summary_tokens": context_optimizer.summary_tokens
            },
            "status": "active"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "redis_connected": redis_cache.is_connected(),
        "database_accessible": True,
        "version": "1.0.0"
    }
