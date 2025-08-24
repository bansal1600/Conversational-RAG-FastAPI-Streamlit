#!/usr/bin/env python3
"""
Environment setup script for production deployment
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_directories():
    """Create necessary directories"""
    project_root = Path(__file__).parent.parent
    
    directories = [
        project_root / "logs",
        project_root / "src" / "database" / "chroma_db",
        project_root / "temp"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if requirements_file.exists():
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✅ Dependencies installed")
    else:
        print("⚠️  requirements.txt not found")

def check_environment():
    """Check environment variables"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
    else:
        print("✅ All required environment variables are set")

def main():
    """Main setup function"""
    print("🚀 Setting up Conversational RAG environment...")
    
    setup_directories()
    install_dependencies()
    check_environment()
    
    print("\n🎉 Environment setup complete!")
    print("Run the server with: python scripts/run_server.py")

if __name__ == "__main__":
    main()
