from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.database.chroma_utils import get_vectorstore
import logging
import time
import hashlib
from src.cache.redis_utils import redis_cache

output_parser = StrOutputParser()

# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Global cache for RAG chains to avoid recreation
_rag_chain_cache = {}

def get_rag_chain(api_key, model="gpt-4o-mini"):
    # Create cache key
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    cache_key = f"{api_key_hash}:{model}"
    
    # Check in-memory cache first (fastest)
    if cache_key in _rag_chain_cache:
        logging.info(f"Memory cache HIT: RAG chain for model {model}")
        return _rag_chain_cache[cache_key]
    
    # Check Redis cache
    cached_config = redis_cache.get_rag_chain_config(api_key, model)
    if cached_config:
        # For now, we'll still rebuild the chain but log the cache hit
        # In a more advanced implementation, we could serialize/deserialize the chain
        logging.info(f"Redis cache HIT: RAG chain config for model {model} (rebuilding chain)")
    else:
        logging.info(f"Cache MISS: Building new RAG chain for model {model}")
    
    # Build the RAG chain
    llm = ChatOpenAI(model=model, api_key=api_key)
    
    # Create vectorstore dynamically with API key
    vectorstore = get_vectorstore(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Cache the chain in memory
    _rag_chain_cache[cache_key] = rag_chain
    
    # Cache configuration in Redis (metadata for monitoring)
    config = {
        "model": model,
        "retriever_k": 2,
        "created_at": str(time.time()),
        "cache_key": cache_key
    }
    redis_cache.set_rag_chain_config(api_key, model, config)
    
    logging.info(f"Created and cached new RAG chain for model {model}")
    return rag_chain

def clear_rag_chain_cache():
    """Clear the in-memory RAG chain cache"""
    global _rag_chain_cache
    _rag_chain_cache.clear()
    logging.info("Cleared RAG chain memory cache")
