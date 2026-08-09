from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from api.deps import get_rag_search, get_nexamind_agent
from api.routes import health_router, sessions_router, rag_router, documents_router, youtube_router, agent_router
from utils.logger import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production FastAPI Backend for NexaMind AI Agent & RAG Platform.",
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
app.include_router(youtube_router)
app.include_router(agent_router)


@app.on_event("startup")
def startup_event():
    """Initializes RAG search engine and AI Agent singletons during startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    get_rag_search()
    get_nexamind_agent()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
