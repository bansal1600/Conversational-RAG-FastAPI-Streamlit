"""
Configuration settings for the Conversational RAG application
"""
import os
from pathlib import Path
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database settings
DATABASE_CONFIG = {
    "sqlite_db_name": "rag_app.db",
    "sqlite_db_path": BASE_DIR / "src" / "database" / "rag_app.db",
    "chroma_db_path": BASE_DIR / "src" / "database" / "chroma_db",
    "max_connections": 10
}

# Redis settings
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "socket_connect_timeout": 5,
    "socket_timeout": 5
}

# Cache settings
CACHE_CONFIG = {
    "chat_history_ttl_hours": 24,
    "rag_chain_ttl_minutes": 30,
    "semantic_cache_ttl_hours": 6,
    "embedding_cache_ttl_hours": 24
}

# Semantic cache settings
SEMANTIC_CACHE_CONFIG = {
    "similarity_threshold": 0.85,
    "max_history_length": 20
}

# Context optimization settings
CONTEXT_CONFIG = {
    "max_context_tokens": 3000,
    "summary_tokens": 500,
    "sliding_window_size": 10,
    "min_recent_messages": 6
}

# Logging settings
LOGGING_CONFIG = {
    "log_file": BASE_DIR / "logs" / "app.log",
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_mode": "a"
}

# API settings
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": True,
    "allowed_file_extensions": ['.pdf', '.docx', '.html']
}

# Environment variables
def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default"""
    return os.getenv(key, default)

# OpenAI settings (from environment)
OPENAI_CONFIG = {
    "api_key": get_env_var("OPENAI_API_KEY"),
    "default_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-ada-002"
}
