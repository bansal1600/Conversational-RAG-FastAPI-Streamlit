# 🤖 Conversational RAG with FastAPI & Streamlit

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI and Streamlit, featuring session-based document isolation, Redis caching, and comprehensive logging.

## ✨ Features

- **🔒 Session-based Document Isolation** - Each user session maintains separate document collections
- **⚡ FastAPI Backend** - High-performance REST API with automatic documentation
- **🎨 Streamlit Frontend** - Interactive web interface for document upload and chat
- **📚 Multi-format Document Support** - PDF, DOCX, TXT files
- **🧠 Intelligent Caching** - Redis-based semantic caching for improved performance
- **📊 Vector Database** - ChromaDB for efficient document retrieval
- **🔍 Conversational Memory** - Context-aware chat with conversation history
- **📝 Comprehensive Logging** - Structured logging with performance monitoring
- **🧪 Test Coverage** - Unit and integration tests included

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│   ChromaDB      │
│                 │    │                 │    │   Vector Store  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │   Redis Cache   │
                       │   + SQLite DB   │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bansal1600/Conversational-RAG-FastAPI-Streamlit.git
   cd Conversational-RAG-FastAPI-Streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Start Redis server**
   ```bash
   redis-server
   ```

5. **Run the application**
   ```bash
   # Start FastAPI backend
   python scripts/run_server.py
   
   # In another terminal, start Streamlit frontend
   streamlit run streamlit_app.py
   ```

6. **Access the application**
   - FastAPI API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

## 📁 Project Structure

```
├── src/
│   ├── cache/              # Redis caching utilities
│   ├── core/               # FastAPI application and main logic
│   ├── database/           # Database connections and utilities
│   ├── models/             # Pydantic models
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── scripts/                # Setup and run scripts
├── tests/                  # Unit and integration tests
├── docs/                   # Sample documents
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Application Configuration
LOG_LEVEL=INFO
```

### Database Setup

The application automatically creates SQLite tables on startup. To reset the database:

```bash
python scripts/reset_database.py
```

## 🔌 API Endpoints

### Document Management
- `POST /upload-doc` - Upload and index documents
- `GET /list-docs` - List uploaded documents by session
- `DELETE /delete-doc/{file_id}` - Delete a document

### Chat Interface
- `POST /chat` - Send messages and get AI responses
- `GET /chat-history/{session_id}` - Retrieve chat history

### Health & Monitoring
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation

## 🎯 Usage Examples

### Upload a Document
```python
import requests

files = {'file': open('document.pdf', 'rb')}
data = {
    'api_key': 'your_openai_key',
    'session_id': 'user_session_123'
}
response = requests.post('http://localhost:8000/upload-doc', files=files, data=data)
```

### Chat with Documents
```python
import requests

payload = {
    'message': 'What is the main topic of the uploaded document?',
    'api_key': 'your_openai_key',
    'session_id': 'user_session_123',
    'model': 'gpt-4o'
}
response = requests.post('http://localhost:8000/chat', json=payload)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📊 Performance Features

- **Semantic Caching**: Reduces API calls by caching similar queries
- **Connection Pooling**: Efficient database connection management
- **Async Processing**: Non-blocking document processing
- **Memory Management**: Optimized vector storage and retrieval

## 🔒 Security

- **API Key Protection**: Environment-based secret management
- **Session Isolation**: User data separation
- **Input Validation**: Pydantic model validation
- **File Type Validation**: Secure file upload handling

## 🚀 Deployment

The application is production-ready with:

- Structured logging
- Error handling
- Health checks
- Performance monitoring
- Scalable architecture
- Docker support (coming soon)

### Environment-specific Configurations

- **Development**: Local SQLite + Redis
- **Production**: Scalable database + Redis Cluster
- **Docker**: Containerized deployment ready

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the frontend
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [OpenAI](https://openai.com/) for language models

## 📞 Support

For support, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for intelligent document processing**
