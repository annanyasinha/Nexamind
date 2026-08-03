# ⚡ RAG Intelligence Platform

A production-grade **Retrieval-Augmented Generation (RAG)** platform engineered with Python, **FAISS**, **Google Gemini Embeddings & LLMs**, **FastAPI**, and a **Streamlit Dashboard**.

---

## 🏗️ Architecture & Project Structure

```
RAG_project/
├── .env.example                  # Environment variable template
├── .gitignore                    # Production gitignore
├── Dockerfile                    # Multi-stage production Docker build
├── docker-compose.yml            # Container orchestration for REST API & UI
├── pyproject.toml                # Package manifest, dependencies & tool configs
├── requirements.txt              # Production dependency specifications
├── README.md                     # Architecture & operational documentation
├── app.py                        # ⭐ Unified application launcher entry point
├── data/                         # Data directory for uploaded documents
│   └── resume.pdf
├── faiss_store/                  # Persistent FAISS vector store index & metadata
├── src/
│   └── rag_platform/             # Core Python package
│       ├── __init__.py
│       ├── config.py             # Centralized Pydantic application settings
│       ├── utils/
│       │   ├── __init__.py
│       │   └── logger.py         # Structured logging configuration
│       ├── core/                 # Business & domain logic
│       │   ├── __init__.py
│       │   ├── document_loader.py# Multi-format document ingestion
│       │   ├── embeddings.py     # Gemini embedding pipeline
│       │   ├── vectorstore.py    # FAISS store builder & search
│       │   ├── search_engine.py  # RAG retrieval & LLM generation engine
│       │   └── session_manager.py# Multi-session conversation state & memory
│       ├── api/                  # FastAPI REST Service Layer
│       │   ├── __init__.py
│       │   ├── main.py           # FastAPI application factory
│       │   ├── deps.py           # Dependency injection & singletons
│       │   ├── schemas.py        # Pydantic request & response models
│       │   └── routes/           # Modular endpoint routers
│       │       ├── health.py     # System health & metrics
│       │       ├── rag.py        # /query & /search RAG endpoints
│       │       ├── sessions.py   # Chat session CRUD endpoints
│       │       └── documents.py  # Document upload & reindexing
│       └── ui/                   # Streamlit Dashboard Layer
│           ├── __init__.py
│           ├── app.py            # Streamlit dashboard interface
│           └── components/
│               └── styles.py     # Glassmorphic CSS themes
└── tests/                        # Automated Pytest Suite
    ├── conftest.py               # Test fixtures
    ├── test_document_loader.py   # Document loading tests
    ├── test_session_manager.py   # Session management tests
    └── test_api_routes.py        # API endpoint integration tests
```

---

## 🌟 Key Features

1. **Enterprise Python Packaging (`src/rag_platform`)**: Modular layout separating configuration, core domain logic, REST API routing, presentation UI, and utilities.
2. **FastAPI REST Backend**:
   - Interactive OpenAPI / Swagger UI documentation at `http://localhost:8000/docs`.
   - Multi-session chat memory management (`/sessions`).
   - RAG query execution with source attribution (`/query`).
   - Raw FAISS vector similarity search (`/search`).
   - Multi-format document uploading & automatic indexing (`/upload`, `/reindex`).
3. **Glassmorphic Streamlit Dashboard**:
   - Interactive multi-session chat interface.
   - Vector Similarity Search Explorer.
   - Drag-and-drop Document Knowledge Base Hub.
   - Real-time API System Health Monitor.
4. **Structured Logging & Configuration**: Centralized Pydantic settings loading `.env` variables and structured logging.
5. **Docker & Container Orchestration**: Multi-stage `Dockerfile` and `docker-compose.yml` for isolated container deployment.
6. **Automated Pytest Suite**: Coverage for document loaders, session state manager, and REST endpoints.

---

## 🛠️ How to Run

### 1. Local Environment Setup

Ensure Python 3.9+ is installed:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure your Google Gemini API key in `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(See `.env.example` for all configurable parameters).*

---

### 2. Launching Services via Application Launcher

- **Start REST Backend Server**:
  ```bash
  python app.py --backend
  ```
  *(Swagger UI available at `http://localhost:8000/docs`)*

- **Start Frontend Streamlit Dashboard**:
  ```bash
  python app.py --frontend
  ```
  *(Dashboard available at `http://localhost:8501`)*

- **Run Direct Query**:
  ```bash
  python app.py --query "Where did Shubham study?"
  ```

---

### 3. Containerized Execution (Docker)

To launch the entire platform using Docker Compose:
```bash
docker-compose up --build
```
- REST Backend Server: `http://localhost:8000`
- Frontend Streamlit UI: `http://localhost:8501`

---

## 🧪 Running Automated Tests

Run the pytest suite to verify document loading, session tracking, and REST API routes:
```bash
python -m pytest tests/ -v
```
