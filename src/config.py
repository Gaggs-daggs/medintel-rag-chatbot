"""
Configuration management for MedIntel RAG Chatbot
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    openai_api_key: str = ""
    gemini_api_key: str = ""
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4-turbo-preview"
    llm_provider: str = "gemini"  # gemini, openai, mistral, qwen, huggingface
    llm_temperature: float = 0.1
    max_tokens: int = 1000
    
    # Retrieval Configuration
    top_k_documents: int = 5
    retrieval_confidence_threshold: float = 0.75
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Vector Store Configuration
    vector_store_path: str = "./data/vector_store"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Application Metadata
    app_name: str = "MedIntel - Medical RAG Chatbot"
    app_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
