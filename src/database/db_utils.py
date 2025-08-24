import sqlite3
from datetime import datetime
import logging
from src.cache.redis_utils import redis_cache
from src.database.connection_manager import connection_manager

DB_NAME = "rag_app.db"

def get_db_connection():
    return connection_manager.get_sqlite_connection()

def create_application_logs():
    with connection_manager.get_sqlite_context() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         session_id TEXT,
                         user_query TEXT,
                         gpt_response TEXT,
                         model TEXT,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

def insert_application_logs(session_id, user_query, gpt_response, model):
    with connection_manager.get_sqlite_context() as conn:
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                     (session_id, user_query, gpt_response, model))
        conn.commit()
    
    # Update Redis cache with new message
    redis_cache.append_to_chat_history(session_id, user_query, gpt_response)

def get_chat_history(session_id):
    # Try to get from Redis cache first
    cached_history = redis_cache.get_chat_history(session_id)
    if cached_history is not None:
        # Convert Redis format to expected format
        messages = []
        for msg in cached_history:
            if msg["role"] == "user":
                messages.append({"role": "human", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "ai", "content": msg["content"]})
        return messages
    
    # Cache miss - get from database
    with connection_manager.get_sqlite_context() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
        messages = []
        cache_format_messages = []
        
        for row in cursor.fetchall():
            messages.extend([
                {"role": "human", "content": row['user_query']},
                {"role": "ai", "content": row['gpt_response']}
            ])
            # Prepare for Redis cache
            cache_format_messages.extend([
                {"role": "user", "content": row['user_query']},
                {"role": "assistant", "content": row['gpt_response']}
            ])
    
    # Cache the result in Redis for future requests
    if cache_format_messages:
        redis_cache.set_chat_history(session_id, cache_format_messages)
    
    return messages

def create_document_store():
    with connection_manager.get_sqlite_context() as conn:
        # Create the table with session_id from the start
        conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT,
                         session_id TEXT,
                         upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

def insert_document_record(filename, session_id=None):
    with connection_manager.get_sqlite_context() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO document_store (filename, session_id) VALUES (?, ?)', (filename, session_id))
        file_id = cursor.lastrowid
        conn.commit()
        return file_id

def delete_document_record(file_id):
    with connection_manager.get_sqlite_context() as conn:
        conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
        conn.commit()
        return True

def get_all_documents(session_id=None):
    with connection_manager.get_sqlite_context() as conn:
        cursor = conn.cursor()
        if session_id:
            cursor.execute('SELECT id, filename, upload_timestamp FROM document_store WHERE session_id = ? ORDER BY upload_timestamp DESC', (session_id,))
        else:
            cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
        documents = cursor.fetchall()
        return [dict(doc) for doc in documents]

# Initialize the database tables
create_application_logs()
create_document_store()
