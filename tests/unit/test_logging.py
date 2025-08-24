#!/usr/bin/env python3
"""
Test script to verify logging is working after Redis implementation
"""
import os
import sys
import logging

# Configure logging exactly like main.py
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Test logging
logging.info("=== LOGGING TEST START ===")
logging.info("Testing if logging works after Redis implementation")

# Import our modules to test they don't break logging
try:
    from redis_utils import redis_cache
    logging.info("Redis utils imported successfully")
    
    from db_utils import get_chat_history
    logging.info("DB utils imported successfully")
    
    from langchain_utils import get_rag_chain
    logging.info("Langchain utils imported successfully")
    
    logging.info("All modules imported without breaking logging")
    
except Exception as e:
    logging.error(f"Error importing modules: {e}")

logging.info("=== LOGGING TEST END ===")
print("Test completed. Check app.log file for results.")
