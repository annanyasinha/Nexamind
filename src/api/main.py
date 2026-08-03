from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from api.deps import get_rag_search
from api.routes import health_router, sessions_router, rag_router, documents_router
from utils.logger import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production FastAPI Backend for Retrieval-Augmented Generation (RAG) powered by FAISS, Gemini, and Session Management.",
    version=settings.VERSION,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health_router)
app.include_router(rag_router)
app.include_router(sessions_router)
app.include_router(documents_router)


@app.on_event("startup")
def startup_event():
    """Initializes the RAG search engine singleton instance during FastAPI server startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    get_rag_search()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
