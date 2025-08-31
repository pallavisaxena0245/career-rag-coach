import os
import logging
from typing import List

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END


def require_packages_ready() -> List[str]:
    missing: List[str] = []
    
    if StateGraph is None:
        missing.append("langgraph")
    
    if ChatOpenAI is None:
        missing.append("langchain-openai")
    
    return missing


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        logger.info("✅ OpenAI API key found in environment")
        logger.info(f"🔑 API key starts with: {api_key[:10]}...")
    else:
        logger.warning("⚠️ No OpenAI API key found in environment")
        logger.info("🔍 Checking for .env file...")
        if os.path.exists(".env"):
            logger.info("📁 .env file exists")
        else:
            logger.info("📁 No .env file found")
    return api_key

