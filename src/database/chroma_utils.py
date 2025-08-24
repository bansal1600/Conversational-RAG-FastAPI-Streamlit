from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
import logging
from langchain_core.documents import Document
from src.database.connection_manager import connection_manager

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

# Function to create and return the vectorstore dynamically using API key
def get_vectorstore(api_key: str) -> Chroma:
    try:
        if api_key:
            # Use connection manager's cached vectorstore
            vectorstore = connection_manager.get_vectorstore(api_key)
            if vectorstore:
                return vectorstore
            else:
                logging.error("Failed to get vectorstore from connection manager")
                return None
        else:
            logging.error("API Key is None")
            return None
    except Exception as e:
        logging.error(f"Error creating vectorstore: {e}")
        return None

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int, api_key: str) -> bool:
    try:
        splits = load_and_split_document(file_path)
        
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        logging.info(f"at before get_vectorstore")
        vectorstore = get_vectorstore(api_key)
        if vectorstore is None:
            logging.error("Failed to create vectorstore - API key may be invalid")
            return False
        vectorstore.add_documents(splits)
        logging.info(f"Successfully indexed {len(splits)} document chunks for file_id {file_id}")
        return True
    except Exception as e:
        logging.error(f"Error indexing document: {e}")
        print(f"Error indexing document: {e}")
        return False

def delete_doc_from_chroma(file_id: int, api_key: str):
    try:
        vectorstore = get_vectorstore(api_key)
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False
