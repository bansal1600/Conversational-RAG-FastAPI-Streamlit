#!/usr/bin/env python3
"""
Unit tests for semantic cache functionality
"""
import sys
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.cache.semantic_cache import SemanticCache, ContextOptimizer

class TestSemanticCache:
    """Test cases for SemanticCache class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.semantic_cache = SemanticCache(similarity_threshold=0.8)
        
    def test_initialization(self):
        """Test semantic cache initialization"""
        assert self.semantic_cache.similarity_threshold == 0.8
        assert self.semantic_cache.max_history_length == 20
        assert self.semantic_cache.embeddings_cache == {}
        
    @patch('src.cache.semantic_cache.redis_cache')
    def test_find_similar_cached_response_no_redis(self, mock_redis):
        """Test behavior when Redis is not connected"""
        mock_redis.is_connected.return_value = False
        
        result = self.semantic_cache.find_similar_cached_response(
            "test query", "test_api_key"
        )
        
        assert result is None
        
    def test_cache_response_no_embedding(self):
        """Test caching when embedding generation fails"""
        with patch.object(self.semantic_cache, '_get_query_embedding', return_value=None):
            result = self.semantic_cache.cache_response(
                "test query", "test response", "test_api_key"
            )
            assert result is False

class TestContextOptimizer:
    """Test cases for ContextOptimizer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = ContextOptimizer(max_context_tokens=100, summary_tokens=50)
        
    def test_initialization(self):
        """Test context optimizer initialization"""
        assert self.optimizer.max_context_tokens == 100
        assert self.optimizer.summary_tokens == 50
        
    def test_estimate_tokens(self):
        """Test token estimation"""
        text = "This is a test message"
        tokens = self.optimizer.estimate_tokens(text)
        expected = len(text) // 4
        assert tokens == expected
        
    def test_compress_chat_history_empty(self):
        """Test compression with empty history"""
        result = self.optimizer.compress_chat_history([], "test_api_key")
        assert result == []
        
    def test_compress_chat_history_under_limit(self):
        """Test compression when under token limit"""
        history = [
            {"role": "human", "content": "Hi"},
            {"role": "ai", "content": "Hello"}
        ]
        result = self.optimizer.compress_chat_history(history, "test_api_key")
        assert result == history
        
    def test_optimize_context_empty(self):
        """Test context optimization with empty history"""
        history, summary = self.optimizer.optimize_context([], "test_api_key")
        assert history == []
        assert summary == ""
        
    def test_summarize_old_context(self):
        """Test old context summarization"""
        old_messages = [
            {"role": "human", "content": "What is AI?"},
            {"role": "ai", "content": "AI is artificial intelligence"},
            {"role": "human", "content": "How does ML work?"}
        ]
        summary = self.optimizer.summarize_old_context(old_messages, "test_api_key")
        assert "What is AI?" in summary
        assert "How does ML work?" in summary

if __name__ == "__main__":
    pytest.main([__file__])
