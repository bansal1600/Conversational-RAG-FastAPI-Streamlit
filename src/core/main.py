import os
import uuid
import logging
import shutil   
# Configure logging FIRST before any other imports
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from redis_utils import redis_cache
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

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
def upload_and_index_document(file: UploadFile = File(...), api_key: str = Form(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"at before insert_document_record")
        file_id = insert_document_record(file.filename)
        logging.info(f"at before index_document_to_chroma")
        success = index_document_to_chroma(temp_file_path, file_id, api_key)
        
        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

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
