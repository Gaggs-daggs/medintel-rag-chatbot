"""
FastAPI backend for MedIntel Medical RAG Chatbot
"""
import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config import get_settings
from src.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    EvaluationRequest,
    EvaluationResponse
)
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.evaluation import RAGASEvaluator

# Global variables
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global vector_store, rag_pipeline
    
    print("=" * 60)
    print("🚀 Starting MedIntel Medical RAG Chatbot")
    print("=" * 60)
    
    try:
        # Initialize vector store
        print("\n📚 Loading vector store...")
        vector_store = VectorStore(
            embedding_model_name=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # Try to load existing vector store
        if os.path.exists(settings.vector_store_path):
            vector_store.load(settings.vector_store_path)
            print(f"✅ Vector store loaded successfully")
            stats = vector_store.get_stats()
            print(f"   - Total chunks: {stats['total_chunks']}")
            print(f"   - Unique documents: {stats['unique_documents']}")
        else:
            print("⚠️  No existing vector store found. Please run the ingestion script first.")
            print(f"   Expected path: {settings.vector_store_path}")
        
        # Initialize RAG pipeline
        print("\n🤖 Initializing RAG pipeline...")
        
        # Determine LLM provider from settings
        llm_provider = settings.llm_provider.lower()
        model_name = None
        api_key = None
        
        if llm_provider == "gemini":
            model_name = settings.llm_model or "gemini-2.0-flash"
            api_key = settings.gemini_api_key
            print(f"   Using Google Gemini ({model_name})")
        elif llm_provider == "mistral":
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            print(f"   Using Mistral 7B (Open Source)")
        elif llm_provider == "qwen":
            model_name = "Qwen/Qwen1.5-7B-Chat"
            print(f"   Using Qwen 1.5 7B Chat (Open Source)")
        else:
            llm_provider = "openai"
            model_name = settings.llm_model
            api_key = settings.openai_api_key
            print(f"   Using OpenAI {model_name}")
        
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            top_k=settings.top_k_documents,
            confidence_threshold=settings.retrieval_confidence_threshold
        )
        
        print("✅ RAG pipeline initialized successfully")
        print("\n" + "=" * 60)
        print(f"🎯 API Server ready at http://{settings.api_host}:{settings.api_port}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        print("   The API will start but may not function correctly.\n")
    
    yield
    
    # Cleanup
    print("\n🛑 Shutting down MedIntel...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A medical Q&A system using RAG that provides accurate, explainable, and citation-backed answers",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MedIntel Medical RAG Chatbot API",
        "version": settings.app_version,
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "evaluate": "/evaluate (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if vector_store and rag_pipeline else "initializing",
        app_name=settings.app_name,
        version=settings.app_version,
        vector_store_loaded=vector_store is not None and len(vector_store.documents) > 0
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the medical RAG system - Hack-A-Cure competition format
    
    Request:
    - **query**: The medical question to answer (required)
    - **top_k**: Number of context snippets to retrieve (required, 1-20)
    
    Response:
    - **answer**: Generated answer with inline citations
    - **contexts**: Array of plain text context snippets
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized. Please check server logs."
        )
    
    if not vector_store or len(vector_store.documents) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not loaded. Please run data ingestion first."
        )
    
    try:
        # Support both 'query' (competition format) and 'question' (legacy)
        question = request.query if request.query else request.question
        
        # Call RAG pipeline
        response = rag_pipeline.query(
            question=question,
            top_k=request.top_k
        )
        
        # Extract plain text contexts from sources
        contexts = [source.excerpt for source in response.sources[:request.top_k]]
        
        # Return competition-compliant format with optional extended fields
        return QueryResponse(
            answer=response.answer,
            contexts=contexts,
            question=question,
            sources=response.sources if request.include_sources else None,
            confidence=response.confidence,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=response.total_time_ms,
            warning=response.warning
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint(request: EvaluationRequest):
    """
    Evaluate the RAG system using RAGAS metrics
    
    - **questions**: List of questions to evaluate
    - **ground_truths**: Optional list of ground truth answers
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized."
        )
    
    try:
        evaluator = RAGASEvaluator(rag_pipeline)
        results = evaluator.evaluate(
            questions=request.questions,
            ground_truths=request.ground_truths
        )
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during evaluation: {str(e)}"
        )


@app.get("/stats", response_model=dict)
async def stats_endpoint():
    """Get vector store statistics"""
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized."
        )
    
    try:
        stats = vector_store.get_stats()
        return {
            "vector_store": stats,
            "rag_config": {
                "llm_model": rag_pipeline.model_name if rag_pipeline else "N/A",
                "llm_provider": rag_pipeline.llm_provider.value if rag_pipeline else "N/A",
                "top_k": settings.top_k_documents,
                "confidence_threshold": settings.retrieval_confidence_threshold,
                "temperature": settings.llm_temperature
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


def start_server():
    """Start the FastAPI server"""
    # Use PORT from environment (Render) or fallback to settings
    port = int(os.getenv("PORT", settings.api_port))
    
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=port,
        reload=False,
        workers=1  # Use 1 worker for development, increase for production
    )


if __name__ == "__main__":
    start_server()
