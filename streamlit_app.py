"""
Streamlit Frontend for Conversational RAG System
Beautiful UI for document upload, chat, and system monitoring
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="Conversational RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"streamlit_{int(time.time())}"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

def make_api_request(endpoint, method="GET", data=None, files=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Conversational RAG System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your OpenAI API key"
        )
        if api_key:
            st.session_state.api_key = api_key
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        
        st.divider()
        
        # System Status
        st.header("📊 System Status")
        
        # Health check
        health_data, health_error = make_api_request("/")
        if health_data:
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API Error: {health_error}")
        
        # Cache stats
        if st.button("🔄 Refresh Stats"):
            cache_data, _ = make_api_request("/cache-stats")
            semantic_data, _ = make_api_request("/semantic-cache-stats")
            connection_data, _ = make_api_request("/connection-stats")
            
            if cache_data:
                st.metric("Redis Status", "Connected" if cache_data.get("cache_status") == "enabled" else "Disconnected")
            
            if semantic_data:
                st.metric("Semantic Cache Entries", semantic_data.get("semantic_cache_entries", 0))
                st.metric("Cached Embeddings", semantic_data.get("cached_embeddings", 0))
            
            if connection_data:
                sqlite_stats = connection_data.get("connection_stats", {}).get("sqlite_pool", {})
                st.metric("DB Connections Created", sqlite_stats.get("connections_created", 0))
                st.metric("DB Connections Reused", sqlite_stats.get("connections_reused", 0))
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Documents", "📈 Analytics", "🔧 Admin"])
    
    with tab1:
        st.header("💬 Chat with Documents")
        
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start chatting.")
            return
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input("Ask a question about your documents...", key="user_input")
            with col2:
                submit_button = st.form_submit_button("Send 🚀")
        
        if submit_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Show thinking spinner
            with st.spinner("🤔 Thinking..."):
                # Make API request
                chat_data = {
                    "question": user_input,
                    "session_id": st.session_state.session_id,
                    "api_key": st.session_state.api_key,
                    "model": model
                }
                
                response_data, error = make_api_request("/chat", method="POST", data=chat_data)
                
                if response_data:
                    assistant_response = response_data.get("answer", "Sorry, I couldn't generate a response.")
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                else:
                    st.error(f"Error: {error}")
            
            st.rerun()
    
    with tab2:
        st.header("📄 Document Management")
        
        # Upload section
        st.subheader("📤 Upload Documents")
        
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your OpenAI API key to upload documents.")
        else:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'docx', 'html'],
                help="Supported formats: PDF, DOCX, HTML"
            )
            
            if uploaded_file and st.button("Upload & Index 📚"):
                with st.spinner("📤 Uploading and indexing document..."):
                    files = {"file": uploaded_file}
                    data = {
                        "api_key": st.session_state.api_key,
                        "session_id": st.session_state.session_id
                    }
                    
                    response_data, error = make_api_request("/upload-doc", method="POST", data=data, files=files)
                    
                    if response_data:
                        st.success(f"✅ {response_data.get('message', 'Document uploaded successfully!')}")
                    else:
                        st.error(f"❌ Upload failed: {error}")
        
        st.divider()
        
        # Document list
        st.subheader("📋 Uploaded Documents")
        
        docs_data, docs_error = make_api_request(f"/list-docs?session_id={st.session_state.session_id}")
        
        if docs_data:
            if docs_data:
                df = pd.DataFrame(docs_data)
                df['upload_timestamp'] = pd.to_datetime(df['upload_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display as table
                st.dataframe(
                    df,
                    column_config={
                        "id": "ID",
                        "filename": "Filename",
                        "upload_timestamp": "Upload Time"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Delete functionality
                if st.session_state.api_key:
                    st.subheader("🗑️ Delete Document")
                    file_to_delete = st.selectbox("Select file to delete", df['filename'].tolist())
                    
                    if st.button("Delete Selected File", type="secondary"):
                        file_id = int(df[df['filename'] == file_to_delete]['id'].iloc[0])
                        delete_data = {
                            "file_id": file_id,
                            "api_key": st.session_state.api_key
                        }
                        
                        response_data, error = make_api_request("/delete-doc", method="POST", data=delete_data)
                        
                        if response_data and "Successfully" in response_data.get("message", ""):
                            st.success("✅ Document deleted successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Delete failed: {error or response_data.get('error', 'Unknown error')}")
            else:
                st.info("📝 No documents uploaded yet.")
        else:
            st.error(f"❌ Failed to load documents: {docs_error}")
    
    with tab3:
        st.header("📈 System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Cache Performance")
            cache_data, _ = make_api_request("/cache-stats")
            if cache_data:
                redis_stats = cache_data.get("redis_stats", {})
                st.metric("Cache Hit Rate", f"{redis_stats.get('hit_rate', 0):.1f}%")
                st.metric("Total Hits", redis_stats.get('hits', 0))
                st.metric("Total Misses", redis_stats.get('misses', 0))
        
        with col2:
            st.subheader("🧠 Semantic Cache")
            semantic_data, _ = make_api_request("/semantic-cache-stats")
            if semantic_data:
                st.metric("Similarity Threshold", f"{semantic_data.get('similarity_threshold', 0):.2f}")
                st.metric("Max History Length", semantic_data.get('max_history_length', 0))
                st.metric("Context Tokens Limit", semantic_data.get('context_optimizer', {}).get('max_context_tokens', 0))
        
        st.divider()
        
        st.subheader("🔗 Connection Statistics")
        connection_data, _ = make_api_request("/connection-stats")
        if connection_data:
            stats = connection_data.get("connection_stats", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**SQLite Pool**")
                sqlite_stats = stats.get("sqlite_pool", {})
                st.write(f"- Max Connections: {sqlite_stats.get('max_connections', 0)}")
                st.write(f"- Created: {sqlite_stats.get('connections_created', 0)}")
                st.write(f"- Reused: {sqlite_stats.get('connections_reused', 0)}")
                st.write(f"- Active: {sqlite_stats.get('active_connections', 0)}")
            
            with col2:
                st.markdown("**ChromaDB Cache**")
                chroma_stats = stats.get("chroma_cache", {})
                st.write(f"- Cached Connections: {chroma_stats.get('cached_connections', 0)}")
                st.write(f"- TTL Minutes: {chroma_stats.get('cache_ttl_minutes', 0)}")
    
    with tab4:
        st.header("🔧 Admin Panel")
        
        if not st.session_state.api_key:
            st.warning("⚠️ API key required for admin functions.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧹 Cache Management")
            if st.button("Clear All Caches"):
                st.info("Cache clearing functionality would be implemented here.")
            
            if st.button("Refresh Connection Pool"):
                st.info("Connection pool refresh would be implemented here.")
        
        with col2:
            st.subheader("📊 System Health")
            if st.button("Run Health Check"):
                health_data, health_error = make_api_request("/")
                cache_data, cache_error = make_api_request("/cache-stats")
                
                if health_data and cache_data:
                    st.success("✅ All systems operational")
                else:
                    st.error("❌ Some systems experiencing issues")
        
        st.divider()
        
        st.subheader("🔄 Session Management")
        st.write(f"**Current Session ID:** `{st.session_state.session_id}`")
        
        if st.button("Start New Session"):
            st.session_state.session_id = f"streamlit_{int(time.time())}"
            st.session_state.chat_history = []
            st.success("✅ New session started!")
            st.rerun()

if __name__ == "__main__":
    main()
