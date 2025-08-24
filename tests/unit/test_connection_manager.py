#!/usr/bin/env python3
"""
Unit tests for connection manager functionality
"""
import sys
import os
import pytest
import tempfile
from unittest.mock import Mock, patch

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection_manager import SQLiteConnectionPool, ChromaConnectionCache, ConnectionManager

class TestSQLiteConnectionPool:
    """Test cases for SQLiteConnectionPool"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.pool = SQLiteConnectionPool(self.temp_db.name, max_connections=5)
        
    def teardown_method(self):
        """Cleanup test fixtures"""
        os.unlink(self.temp_db.name)
        
    def test_initialization(self):
        """Test connection pool initialization"""
        assert self.pool.db_path == self.temp_db.name
        assert self.pool.max_connections == 5
        
    def test_get_connection(self):
        """Test getting a connection from pool"""
        conn = self.pool.get_connection()
        assert conn is not None
        
        # Test connection reuse
        conn2 = self.pool.get_connection()
        assert conn is conn2
        
    def test_connection_context(self):
        """Test connection context manager"""
        with self.pool.get_connection_context() as conn:
            assert conn is not None
            conn.execute("CREATE TABLE test (id INTEGER)")

class TestChromaConnectionCache:
    """Test cases for ChromaConnectionCache"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cache = ChromaConnectionCache()
        
    def test_initialization(self):
        """Test cache initialization"""
        assert self.cache.cache_ttl_minutes == 30
        assert len(self.cache._connections) == 0
        
    @patch('src.database.connection_manager.OpenAIEmbeddings')
    @patch('src.database.connection_manager.Chroma')
    def test_get_vectorstore_success(self, mock_chroma, mock_embeddings):
        """Test successful vectorstore creation"""
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        result = self.cache.get_vectorstore("test_api_key")
        
        assert result == mock_vectorstore
        mock_embeddings.assert_called_once_with(openai_api_key="test_api_key")
        
    def test_get_vectorstore_no_api_key(self):
        """Test vectorstore creation with no API key"""
        result = self.cache.get_vectorstore(None)
        assert result is None
        
    def test_clear_cache(self):
        """Test cache clearing"""
        self.cache._connections["test"] = Mock()
        assert len(self.cache._connections) == 1
        
        self.cache.clear_cache()
        assert len(self.cache._connections) == 0
        
    def test_get_cache_stats(self):
        """Test cache statistics"""
        self.cache._connections["test1"] = Mock()
        self.cache._connections["test2"] = Mock()
        
        stats = self.cache.get_cache_stats()
        assert stats["cached_connections"] == 2
        assert "test1" in stats["cache_keys"]
        assert "test2" in stats["cache_keys"]

class TestConnectionManager:
    """Test cases for ConnectionManager"""
    
    @patch('src.database.connection_manager.DATABASE_CONFIG')
    def setup_method(self, mock_config):
        """Setup test fixtures"""
        mock_config.__getitem__.side_effect = lambda key: {
            "sqlite_db_path": ":memory:",
            "max_connections": 10,
            "chroma_db_path": "/tmp/test_chroma"
        }[key]
        
        self.manager = ConnectionManager()
        
    def test_initialization(self):
        """Test connection manager initialization"""
        assert self.manager.sqlite_pool is not None
        assert self.manager.chroma_cache is not None
        
    def test_get_sqlite_connection(self):
        """Test getting SQLite connection"""
        conn = self.manager.get_sqlite_connection()
        assert conn is not None
        
    def test_get_connection_stats(self):
        """Test getting connection statistics"""
        stats = self.manager.get_connection_stats()
        assert "sqlite_pool" in stats
        assert "chroma_cache" in stats
        assert "db_path" in stats["sqlite_pool"]
        assert "max_connections" in stats["sqlite_pool"]

if __name__ == "__main__":
    pytest.main([__file__])
