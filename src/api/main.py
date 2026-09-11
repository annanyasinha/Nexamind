from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.rate_limit import limiter

from api.deps import get_nexamind_agent, get_rag_search
from api.routes import (
    agent_router,
    documents_router,
    health_router,
    rag_router,
    sessions_router,
    youtube_router,
    github_router,
)
from config import settings
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes RAG search engine and AI Agent singletons during startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    get_rag_search()
    get_nexamind_agent()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production FastAPI Backend for NexaMind AI Agent & RAG Platform.",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
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
app.include_router(github_router)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
