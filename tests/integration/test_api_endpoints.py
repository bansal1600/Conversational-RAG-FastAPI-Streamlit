#!/usr/bin/env python3
"""
Integration tests for API endpoints
"""
import sys
import os
import pytest
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.app import app

class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(app)
        
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "redis_connected" in data
        assert "database_accessible" in data
        assert "version" in data
        
    @patch('src.core.app.redis_cache')
    def test_cache_stats_endpoint(self, mock_redis):
        """Test cache stats endpoint"""
        mock_redis.get_cache_stats.return_value = {"connected": True}
        mock_redis.is_connected.return_value = True
        
        response = self.client.get("/cache-stats")
        assert response.status_code == 200
        data = response.json()
        assert "redis_stats" in data
        assert "cache_status" in data
        
    @patch('src.core.app.connection_manager')
    def test_connection_stats_endpoint(self, mock_manager):
        """Test connection stats endpoint"""
        mock_manager.get_connection_stats.return_value = {
            "sqlite_pool": {"connections": 1},
            "chroma_cache": {"cached_connections": 0}
        }
        
        response = self.client.get("/connection-stats")
        assert response.status_code == 200
        data = response.json()
        assert "connection_stats" in data
        assert data["status"] == "optimized"
        
    @patch('src.core.app.redis_cache')
    def test_semantic_cache_stats_endpoint(self, mock_redis):
        """Test semantic cache stats endpoint"""
        mock_redis.is_connected.return_value = True
        mock_redis.redis_client.keys.side_effect = [
            ["semantic_cache:key1", "semantic_cache:key2"],
            ["embedding:key1"]
        ]
        
        response = self.client.get("/semantic-cache-stats")
        assert response.status_code == 200
        data = response.json()
        assert "semantic_cache_entries" in data
        assert "cached_embeddings" in data
        assert "similarity_threshold" in data
        
    def test_list_documents_endpoint(self):
        """Test list documents endpoint"""
        with patch('src.core.app.get_all_documents') as mock_get_docs:
            mock_get_docs.return_value = []
            
            response = self.client.get("/list-docs")
            assert response.status_code == 200
            assert response.json() == []

if __name__ == "__main__":
    pytest.main([__file__])
