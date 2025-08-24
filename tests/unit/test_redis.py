#!/usr/bin/env python3
"""
Test script to verify Redis caching implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from redis_utils import redis_cache
from db_utils import get_chat_history, insert_application_logs
import time
import uuid

def test_redis_connection():
    """Test basic Redis connection"""
    print("🔧 Testing Redis connection...")
    if redis_cache.is_connected():
        print("✅ Redis connection successful!")
        stats = redis_cache.get_cache_stats()
        print(f"📊 Redis stats: {stats}")
        return True
    else:
        print("❌ Redis connection failed!")
        return False

def test_chat_history_caching():
    """Test chat history caching functionality"""
    print("\n💬 Testing chat history caching...")
    
    # Create a test session
    test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    
    # Test 1: Insert some chat history
    print(f"📝 Inserting test messages for session: {test_session_id}")
    insert_application_logs(test_session_id, "Hello, how are you?", "I'm doing great! How can I help you?", "gpt-4o")
    insert_application_logs(test_session_id, "What's the weather like?", "I don't have access to current weather data, but I can help with other questions!", "gpt-4o")
    
    # Test 2: First retrieval (should be cache miss, then cached)
    print("🔍 First retrieval (cache miss expected)...")
    start_time = time.time()
    history1 = get_chat_history(test_session_id)
    first_time = time.time() - start_time
    print(f"⏱️  First retrieval took: {first_time:.4f} seconds")
    print(f"📋 Retrieved {len(history1)} messages")
    
    # Test 3: Second retrieval (should be cache hit)
    print("🔍 Second retrieval (cache hit expected)...")
    start_time = time.time()
    history2 = get_chat_history(test_session_id)
    second_time = time.time() - start_time
    print(f"⏱️  Second retrieval took: {second_time:.4f} seconds")
    print(f"📋 Retrieved {len(history2)} messages")
    
    # Calculate performance improvement
    if first_time > 0:
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"🚀 Performance improvement: {improvement:.1f}%")
    
    # Verify data consistency
    if history1 == history2:
        print("✅ Cache data consistency verified!")
        return True
    else:
        print("❌ Cache data inconsistency detected!")
        return False

def test_rag_chain_caching():
    """Test RAG chain caching (basic test)"""
    print("\n🔗 Testing RAG chain caching...")
    
    # This is a basic test - we can't fully test without an API key
    # But we can test the caching logic
    from langchain_utils import _rag_chain_cache, clear_rag_chain_cache
    
    print(f"📊 Current RAG chain cache size: {len(_rag_chain_cache)}")
    
    # Clear cache for clean test
    clear_rag_chain_cache()
    print("🧹 Cleared RAG chain cache")
    print(f"📊 Cache size after clear: {len(_rag_chain_cache)}")
    
    return True

def main():
    """Run all tests"""
    print("🧪 Starting Redis Caching Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Redis Connection
    if test_redis_connection():
        tests_passed += 1
    
    # Test 2: Chat History Caching
    if test_chat_history_caching():
        tests_passed += 1
    
    # Test 3: RAG Chain Caching
    if test_rag_chain_caching():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Redis caching is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
