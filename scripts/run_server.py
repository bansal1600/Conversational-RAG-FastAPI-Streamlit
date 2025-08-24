#!/usr/bin/env python3
"""
Production server runner script
"""
import sys
import os
import uvicorn

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.settings import API_CONFIG

def run_server():
    """Run the FastAPI server with production settings"""
    uvicorn.run(
        "src.core.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
