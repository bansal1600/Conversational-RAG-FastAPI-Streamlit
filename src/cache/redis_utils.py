import redis
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import timedelta
import logging

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logging.info("Redis connection established successfully")
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False

    def get_chat_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """Get cached chat history for a session"""
        if not self.is_connected():
            return None
        
        try:
            key = f"chat_history:{session_id}"
            cached_data = self.redis_client.get(key)
            if cached_data:
                logging.info(f"Cache HIT: Chat history for session {session_id}")
                return json.loads(cached_data)
            else:
                logging.info(f"Cache MISS: Chat history for session {session_id}")
                return None
        except Exception as e:
            logging.error(f"Error getting chat history from cache: {e}")
            return None

    def set_chat_history(self, session_id: str, chat_history: List[Dict[str, str]], ttl_hours: int = 24):
        """Cache chat history for a session"""
        if not self.is_connected():
            return False
        
        try:
            key = f"chat_history:{session_id}"
            self.redis_client.setex(
                key, 
                timedelta(hours=ttl_hours), 
                json.dumps(chat_history)
            )
            logging.info(f"Cached chat history for session {session_id}")
            return True
        except Exception as e:
            logging.error(f"Error caching chat history: {e}")
            return False

    def append_to_chat_history(self, session_id: str, user_query: str, ai_response: str):
        """Append new message to cached chat history"""
        if not self.is_connected():
            return False
        
        try:
            # Get current history
            current_history = self.get_chat_history(session_id) or []
            
            # Add new messages
            current_history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": ai_response}
            ])
            
            # Update cache
            return self.set_chat_history(session_id, current_history)
        except Exception as e:
            logging.error(f"Error appending to chat history: {e}")
            return False

    def get_rag_chain_config(self, api_key: str, model: str) -> Optional[Dict[str, Any]]:
        """Get cached RAG chain configuration"""
        if not self.is_connected():
            return None
        
        try:
            # Create hash of API key for security
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            key = f"rag_chain:{api_key_hash}:{model}"
            
            cached_data = self.redis_client.get(key)
            if cached_data:
                logging.info(f"Cache HIT: RAG chain config for model {model}")
                return json.loads(cached_data)
            else:
                logging.info(f"Cache MISS: RAG chain config for model {model}")
                return None
        except Exception as e:
            logging.error(f"Error getting RAG chain config from cache: {e}")
            return None

    def set_rag_chain_config(self, api_key: str, model: str, config: Dict[str, Any], ttl_minutes: int = 30):
        """Cache RAG chain configuration"""
        if not self.is_connected():
            return False
        
        try:
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            key = f"rag_chain:{api_key_hash}:{model}"
            
            self.redis_client.setex(
                key,
                timedelta(minutes=ttl_minutes),
                json.dumps(config)
            )
            logging.info(f"Cached RAG chain config for model {model}")
            return True
        except Exception as e:
            logging.error(f"Error caching RAG chain config: {e}")
            return False

    def invalidate_chat_history(self, session_id: str):
        """Remove chat history from cache"""
        if not self.is_connected():
            return False
        
        try:
            key = f"chat_history:{session_id}"
            self.redis_client.delete(key)
            logging.info(f"Invalidated chat history for session {session_id}")
            return True
        except Exception as e:
            logging.error(f"Error invalidating chat history: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.is_connected():
            return {"connected": False}
        
        try:
            info = self.redis_client.info()
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            return {
                "connected": True,
                "used_memory_human": info.get('used_memory_human', 'N/A'),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "keyspace_hits": hits,
                "keyspace_misses": misses
            }
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return {"connected": False, "error": str(e)}

# Global Redis cache instance
redis_cache = RedisCache()
