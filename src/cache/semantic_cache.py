import hashlib
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.cache.redis_utils import redis_cache
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import base64

class SemanticCache:
    """Advanced semantic caching system for RAG queries"""
    
    def __init__(self, similarity_threshold: float = 0.85, max_history_length: int = 20):
        self.similarity_threshold = similarity_threshold
        self.max_history_length = max_history_length
        self.embeddings_cache = {}
        self._embedding_model = None
        
    def _get_embedding_model(self, api_key: str):
        """Get or create embedding model"""
        if not self._embedding_model:
            self._embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        return self._embedding_model
    
    def _get_query_embedding(self, query: str, api_key: str) -> Optional[np.ndarray]:
        """Get embedding for a query with caching"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check local cache first
        if query_hash in self.embeddings_cache:
            return self.embeddings_cache[query_hash]
        
        # Check Redis cache
        redis_key = f"embedding:{query_hash}"
        cached_embedding = redis_cache.redis_client.get(redis_key) if redis_cache.is_connected() else None
        
        if cached_embedding:
            try:
                embedding = pickle.loads(base64.b64decode(cached_embedding))
                self.embeddings_cache[query_hash] = embedding
                logging.info(f"Embedding cache HIT for query hash {query_hash[:8]}")
                return embedding
            except Exception as e:
                logging.error(f"Error loading cached embedding: {e}")
        
        # Generate new embedding
        try:
            embedding_model = self._get_embedding_model(api_key)
            embedding = np.array(embedding_model.embed_query(query))
            
            # Cache locally and in Redis
            self.embeddings_cache[query_hash] = embedding
            if redis_cache.is_connected():
                encoded_embedding = base64.b64encode(pickle.dumps(embedding)).decode()
                redis_cache.redis_client.setex(redis_key, timedelta(hours=24), encoded_embedding)
            
            logging.info(f"Generated new embedding for query hash {query_hash[:8]}")
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
    
    def find_similar_cached_response(self, query: str, api_key: str, context_hash: str = None) -> Optional[Dict[str, Any]]:
        """Find semantically similar cached responses"""
        query_embedding = self._get_query_embedding(query, api_key)
        if query_embedding is None:
            return None
        
        # Search for similar queries in Redis
        if not redis_cache.is_connected():
            return None
        
        try:
            # Get all semantic cache keys
            pattern = "semantic_cache:*"
            cache_keys = redis_cache.redis_client.keys(pattern)
            
            best_match = None
            best_similarity = 0.0
            
            for key in cache_keys:
                cached_data = redis_cache.redis_client.get(key)
                if cached_data:
                    try:
                        cache_entry = json.loads(cached_data)
                        cached_embedding = np.array(cache_entry['embedding'])
                        
                        # Calculate cosine similarity
                        similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
                        
                        if similarity > self.similarity_threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                'response': cache_entry['response'],
                                'similarity': similarity,
                                'original_query': cache_entry['query'],
                                'timestamp': cache_entry['timestamp']
                            }
                    except Exception as e:
                        logging.error(f"Error processing cache entry {key}: {e}")
            
            if best_match:
                logging.info(f"Semantic cache HIT: {best_similarity:.3f} similarity for query: {query[:50]}...")
                return best_match
            else:
                logging.info(f"Semantic cache MISS for query: {query[:50]}...")
                return None
                
        except Exception as e:
            logging.error(f"Error searching semantic cache: {e}")
            return None
    
    def cache_response(self, query: str, response: str, api_key: str, context_hash: str = None, ttl_hours: int = 6):
        """Cache a query-response pair with semantic indexing"""
        query_embedding = self._get_query_embedding(query, api_key)
        if query_embedding is None:
            return False
        
        try:
            # Create cache entry
            cache_entry = {
                'query': query,
                'response': response,
                'embedding': query_embedding.tolist(),
                'context_hash': context_hash,
                'timestamp': datetime.now().isoformat(),
                'api_key_hash': hashlib.sha256(api_key.encode()).hexdigest()[:16]
            }
            
            # Generate cache key
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cache_key = f"semantic_cache:{query_hash}"
            
            # Store in Redis
            if redis_cache.is_connected():
                redis_cache.redis_client.setex(
                    cache_key,
                    timedelta(hours=ttl_hours),
                    json.dumps(cache_entry)
                )
                logging.info(f"Cached semantic response for query hash {query_hash[:8]}")
                return True
            
        except Exception as e:
            logging.error(f"Error caching semantic response: {e}")
        
        return False

class ContextOptimizer:
    """Optimize chat context to manage token limits"""
    
    def __init__(self, max_context_tokens: int = 3000, summary_tokens: int = 500):
        self.max_context_tokens = max_context_tokens
        self.summary_tokens = summary_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def compress_chat_history(self, chat_history: List[Dict[str, str]], api_key: str) -> List[Dict[str, str]]:
        """Compress chat history using sliding window and summarization"""
        if not chat_history:
            return chat_history
        
        total_tokens = sum(self.estimate_tokens(msg.get('content', '')) for msg in chat_history)
        
        if total_tokens <= self.max_context_tokens:
            return chat_history
        
        logging.info(f"Compressing chat history: {total_tokens} tokens -> target: {self.max_context_tokens}")
        
        # Keep recent messages (sliding window)
        recent_messages = chat_history[-10:]  # Keep last 10 messages
        recent_tokens = sum(self.estimate_tokens(msg.get('content', '')) for msg in recent_messages)
        
        if recent_tokens <= self.max_context_tokens:
            return recent_messages
        
        # If still too long, keep only the most recent exchanges
        return chat_history[-6:]  # Keep last 3 exchanges (6 messages)
    
    def summarize_old_context(self, old_messages: List[Dict[str, str]], api_key: str) -> str:
        """Summarize older messages to preserve context"""
        if not old_messages:
            return ""
        
        # Create a simple summary of key topics
        user_queries = [msg['content'] for msg in old_messages if msg.get('role') == 'human']
        
        if len(user_queries) <= 2:
            return f"Previous topics discussed: {', '.join(user_queries[:2])}"
        
        return f"Previous conversation covered {len(user_queries)} topics including: {', '.join(user_queries[:3])}..."
    
    def optimize_context(self, chat_history: List[Dict[str, str]], api_key: str) -> Tuple[List[Dict[str, str]], str]:
        """Optimize context and return compressed history + summary"""
        if not chat_history:
            return chat_history, ""
        
        total_tokens = sum(self.estimate_tokens(msg.get('content', '')) for msg in chat_history)
        
        if total_tokens <= self.max_context_tokens:
            return chat_history, ""
        
        # Split into old and recent messages
        split_point = max(0, len(chat_history) - 10)
        old_messages = chat_history[:split_point]
        recent_messages = chat_history[split_point:]
        
        # Summarize old context
        summary = self.summarize_old_context(old_messages, api_key)
        
        # Compress recent messages
        compressed_recent = self.compress_chat_history(recent_messages, api_key)
        
        logging.info(f"Context optimization: {len(chat_history)} -> {len(compressed_recent)} messages + summary")
        
        return compressed_recent, summary

# Global instances
semantic_cache = SemanticCache()
context_optimizer = ContextOptimizer()
