"""
Connection Manager for SQLite and ChromaDB connections with pooling and caching.
Provides thread-safe connection management and resource optimization.
"""

import sqlite3
import threading
import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config.settings import DATABASE_CONFIG
import os

logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool for better performance."""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._local = threading.local()
        self._pool = []
        self._pool_lock = threading.Lock()
        self._stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'active_connections': 0
        }
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection for the current thread."""
        # Check if current thread already has a connection
        if hasattr(self._local, 'connection') and self._local.connection:
            self._stats['connections_reused'] += 1
            return self._local.connection
        
        # Try to get from pool
        with self._pool_lock:
            if self._pool:
                conn = self._pool.pop()
                self._local.connection = conn
                self._stats['connections_reused'] += 1
                self._stats['active_connections'] += 1
                return conn
        
        # Create new connection
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._local.connection = conn
        self._stats['connections_created'] += 1
        self._stats['active_connections'] += 1
        logger.debug(f"Created new SQLite connection. Total: {self._stats['connections_created']}")
        return conn
    
    @contextmanager
    def get_connection_context(self):
        """Context manager for automatic connection cleanup."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            # Connection stays in thread-local storage for reuse
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'db_path': self.db_path,
            'max_connections': self.max_connections,
            **self._stats
        }

class ChromaConnectionCache:
    """Cache for ChromaDB vectorstore instances to avoid expensive recreations."""
    
    def __init__(self, cache_ttl_minutes: int = 30):
        self.cache_ttl_minutes = cache_ttl_minutes
        self._connections = {}
        self._lock = threading.Lock()
    
    def get_vectorstore(self, api_key: str) -> Optional[Chroma]:
        """Get or create vectorstore instance."""
        if not api_key:
            return None
            
        cache_key = f"chroma_{hash(api_key)}"
        
        with self._lock:
            # Check if we have a cached connection
            if cache_key in self._connections:
                vectorstore, timestamp = self._connections[cache_key]
                # Check if cache is still valid
                if time.time() - timestamp < (self.cache_ttl_minutes * 60):
                    logger.debug(f"Using cached ChromaDB connection")
                    return vectorstore
                else:
                    # Remove expired cache
                    del self._connections[cache_key]
        
        # Create new vectorstore
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            chroma_path = str(DATABASE_CONFIG["chroma_db_path"])
            os.makedirs(chroma_path, exist_ok=True)
            vectorstore = Chroma(
                persist_directory=chroma_path,
                embedding_function=embeddings
            )
            
            # Cache the new connection
            with self._lock:
                self._connections[cache_key] = (vectorstore, time.time())
            
            logger.debug(f"Created new ChromaDB connection")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB connection: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached connections."""
        with self._lock:
            self._connections.clear()
            logger.info("Cleared ChromaDB connection cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cached_connections": len(self._connections),
                "cache_ttl_minutes": self.cache_ttl_minutes,
                "cache_keys": list(self._connections.keys())
            }

class ConnectionManager:
    """Main connection manager that coordinates SQLite and ChromaDB connections."""
    
    def __init__(self):
        # Ensure database path exists and convert to string
        db_path = str(DATABASE_CONFIG["sqlite_db_path"])
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.sqlite_pool = SQLiteConnectionPool(
            db_path,
            DATABASE_CONFIG.get("max_connections", 10)
        )
        self.chroma_cache = ChromaConnectionCache()
        logger.info("Connection manager initialized")
    
    def get_sqlite_connection(self) -> sqlite3.Connection:
        """Get SQLite connection from pool."""
        return self.sqlite_pool.get_connection()
    
    def get_sqlite_context(self):
        """Get SQLite connection context manager."""
        return self.sqlite_pool.get_connection_context()
    
    def get_vectorstore(self, api_key: str) -> Optional[Chroma]:
        """Get ChromaDB vectorstore instance."""
        return self.chroma_cache.get_vectorstore(api_key)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics."""
        return {
            "sqlite_pool": self.sqlite_pool.get_stats(),
            "chroma_cache": self.chroma_cache.get_cache_stats()
        }
    
    def cleanup(self):
        """Cleanup all connections and caches."""
        self.chroma_cache.clear_cache()
        logger.info("Connection manager cleanup completed")

# Global connection manager instance
connection_manager = ConnectionManager()
