# ⚡ NexaMind — Multi-Source RAG Intelligence Platform

NexaMind is a **multi-source Retrieval-Augmented Generation (RAG) platform** built using **Python, FastAPI, Streamlit, Google Gemini, FAISS, LangChain, and MongoDB Atlas**.

It enables users to ask natural-language questions over multiple knowledge sources including:

- 📄 Uploaded documents
- 📹 YouTube transcripts
- 💻 Public GitHub repositories
- 🌐 Live web information

Instead of relying only on an LLM's internal knowledge, NexaMind retrieves relevant source content first and provides it to the language model to generate **context-aware and source-grounded responses**.

---

## 🌟 Key Features

### 📄 Multi-Format Document RAG

NexaMind supports document ingestion for:

- PDF
- TXT
- CSV
- XLSX
- DOCX
- JSON

Uploaded documents are parsed, chunked, converted into vector embeddings, and stored in FAISS for semantic retrieval.

The document pipeline also supports:

- Incremental indexing for new files
- SHA-256 based content deduplication
- Automatic full index rebuild when an existing file changes
- File-size and extension validation
- Source metadata preservation

---

### 💻 GitHub Repository RAG

NexaMind can perform semantic question answering over public GitHub repositories.

The GitHub pipeline:

```text
GitHub Repository URL
        ↓
GitHub REST API
        ↓
Repository Metadata
        ↓
Latest Commit SHA
        ↓
Recursive Git Tree
        ↓
Filter Supported Source Files
        ↓
Fetch Git Blobs
        ↓
Decode File Content
        ↓
Create Documents + Provenance
        ↓
Chunking
        ↓
Gemini Embeddings
        ↓
Repository-Specific FAISS Index
        ↓
Semantic Retrieval
        ↓
Gemini Grounded Answer
```

Each repository gets its own FAISS vector store, preventing retrieval results from unrelated repositories from mixing together.

Source metadata includes information such as:

- Repository name
- File path
- File name
- Language
- Commit SHA
- File SHA
- Commit-pinned GitHub source URL

---

## 🔄 GitHub Commit-SHA Cache Invalidation

Repository indexing uses **commit-SHA-based cache validation**.

```text
User provides GitHub repository
            ↓
Fetch latest commit SHA
            ↓
Read cached manifest.json
            ↓
Compare commit SHAs
        /             \
     Same            Changed
      ↓                 ↓
Reuse FAISS         Rebuild FAISS
index               repository index
```

The repository cache contains:

```text
github_faiss_store/
└── owner__repo/
    ├── faiss.index
    ├── metadata.pkl
    └── manifest.json
```

This prevents unnecessary re-embedding when the repository has not changed.

---

## 📹 YouTube RAG

NexaMind can retrieve YouTube transcripts and create a dedicated knowledge base for each video.

The pipeline is:

```text
YouTube URL
    ↓
Video ID Extraction
    ↓
YouTube Transcript API
    ↓
Timestamped Transcript
    ↓
Chunking
    ↓
Gemini Embeddings
    ↓
Per-Video FAISS Index
    ↓
Semantic Retrieval
    ↓
Grounded Answer
```

Timestamp metadata is preserved so retrieved information can be associated with the relevant part of the video.

---

## 🌐 Live Web Search

NexaMind also supports live web retrieval for questions requiring current or external information.

Unlike documents, YouTube videos, and GitHub repositories, live web information does not need to be permanently stored in the main FAISS knowledge base.

This allows the system to combine:

```text
Persistent Knowledge
+
Live Information
```

within the same AI assistant.

---

## 🤖 Intelligent Agent Routing

NexaMind includes an AI agent capable of selecting the appropriate retrieval tool depending on the user's query.

Available tools include:

```text
Document RAG
YouTube RAG
GitHub RAG
Live Web Search
```

The agent uses a hybrid routing strategy consisting of:

```text
User Query
    ↓
Deterministic Source Detection
    ↓
LLM-Based JSON Planning
    ↓
Heuristic Fallback
    ↓
Selected Tool(s)
    ↓
Retrieved Evidence
    ↓
Gemini Response Synthesis
```

This enables different data sources to use their own ingestion, retrieval, freshness, and provenance strategies while remaining accessible through one interface.

---

## 🧠 RAG Pipeline

The general NexaMind retrieval pipeline is:

```text
Source
  ↓
Document Parsing
  ↓
Recursive Character Chunking
  ↓
Gemini Embeddings
  ↓
L2 Vector Normalization
  ↓
FAISS IndexFlatIP
  ↓
Cosine Similarity Retrieval
  ↓
Relevant Context
  ↓
Gemini LLM
  ↓
Grounded Response + Sources
```

### Chunking Configuration

Default configuration:

```text
Chunk Size    : 1000 characters
Chunk Overlap : 200 characters
```

NexaMind uses recursive character-based chunking to preserve useful textual boundaries wherever possible.

---

## 🔍 Vector Search

NexaMind uses **FAISS IndexFlatIP** for dense-vector retrieval.

Before vectors are inserted into FAISS:

```text
Embedding
   ↓
L2 Normalization
   ↓
IndexFlatIP
```

Query embeddings are also normalized.

Because both stored vectors and query vectors are unit-normalized:

```text
Inner Product ≈ Cosine Similarity
```

This allows NexaMind to retrieve chunks that are semantically similar to the user's query.

---

## 🗄️ Persistent Chat Sessions with MongoDB Atlas

MongoDB Atlas is used for **persistent conversation storage**.

It stores:

```text
sessions
messages
```

Each message can contain:

- Session ID
- Role
- Message content
- Sources
- Creation timestamp

This enables conversation history to survive application restarts.

MongoDB is used as the application's persistent session database and is separate from FAISS, which is responsible for vector retrieval.

---

## ⚡ FAISS Persistence

FAISS indexes are persisted to the filesystem rather than rebuilt for every request.

Typical storage:

```text
faiss_store/
├── faiss.index
└── metadata.pkl

youtube_faiss_store/
└── <video_id>/
    ├── faiss.index
    └── metadata.pkl

github_faiss_store/
└── <owner>__<repo>/
    ├── faiss.index
    ├── metadata.pkl
    └── manifest.json
```

This reduces repeated embedding generation and improves response latency.

---

## 🛡️ API Rate Limiting

NexaMind implements **IP-based API rate limiting using SlowAPI**.

Expensive operations have stricter limits because they may involve external APIs, embedding generation, LLM inference, or FAISS index construction.

| Endpoint | Rate Limit |
|---|---:|
| `/reindex` | 2/minute |
| `/github/index` | 3/minute |
| `/upload` | 5/minute |
| Document delete operations | 5/minute |
| `/youtube/transcript` | 5/minute |
| `/query/stream` | 10/minute |
| `/agent/query` | 15/minute |
| `/query` | 20/minute |
| `/github/query` | 20/minute |
| `/search` | 30/minute |

If a client exceeds a configured limit, the API returns:

```text
HTTP 429 Too Many Requests
```

The current single-instance implementation maintains rate-limit counters in application memory.

---

## 📡 Server-Sent Events Streaming

NexaMind supports streaming RAG responses using **Server-Sent Events (SSE)**.

```text
Client
  ↓
POST /query/stream
  ↓
FastAPI
  ↓
Retrieve Context
  ↓
Gemini Generation
  ↓
SSE Token Stream
  ↓
Client
```

The streaming endpoint can send:

```text
sources
tokens
done
```

events while the answer is being generated.

---

## 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │  Streamlit  │
                    │  Frontend   │
                    └──────┬──────┘
                           │ HTTP / SSE
                           ▼
                    ┌─────────────┐
                    │   FastAPI   │
                    │   Backend   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Rate Limiter│
                    │   SlowAPI   │
                    └──────┬──────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ NexaMind Agent│
                   └───────┬───────┘
                           │
             ┌─────────────┼──────────────┐
             │             │              │
             ▼             ▼              ▼
       Document RAG   YouTube RAG     GitHub RAG
             │             │              │
             └─────────────┼──────────────┤
                           │              │
                           ▼              ▼
                       FAISS          Live Web
                           │              │
                           └──────┬───────┘
                                  │
                                  ▼
                           Gemini LLM
                                  │
                                  ▼
                    Grounded Answer + Sources


                    MongoDB Atlas
                         │
                         └── Sessions & Chat History
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Frontend | Streamlit |
| Backend | FastAPI |
| API Validation | Pydantic |
| RAG Framework | LangChain |
| Embeddings | Google Gemini Embeddings |
| LLM | Google Gemini |
| Vector Database / Search | FAISS |
| Similarity | Cosine Similarity |
| Database | MongoDB Atlas |
| GitHub Integration | GitHub REST API |
| YouTube Integration | YouTube Transcript API |
| Web Retrieval | Live Web Search |
| API Rate Limiting | SlowAPI |
| Streaming | Server-Sent Events |
| Containerization | Docker |
| Local Orchestration | Docker Compose |
| Testing | Pytest |
| Deployment | Render |

---

## 📂 Project Structure

```text
Nexamind/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
│
├── data/
│
├── faiss_store/
├── youtube_faiss_store/
├── github_faiss_store/
│
├── src/
│   │
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── deps.py
│   │   ├── schemas.py
│   │   ├── rate_limit.py
│   │   │
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── rag.py
│   │       ├── documents.py
│   │       ├── sessions.py
│   │       ├── youtube.py
│   │       ├── github.py
│   │       └── agent.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── database.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── github_loader.py
│   │   ├── prompt.py
│   │   ├── search_engine.py
│   │   ├── session_manager.py
│   │   ├── tools.py
│   │   ├── vectorstore.py
│   │   └── youtube_loader.py
│   │
│   ├── ui/
│   │   ├── app.py
│   │   └── components/
│   │       └── styles.py
│   │
│   └── utils/
│       └── logger.py
│
└── tests/
```

---

## 🔌 Important API Endpoints

### RAG

```text
POST /query
POST /query/stream
POST /search
```

### AI Agent

```text
POST /agent/query
```

### Documents

```text
GET    /documents
POST   /upload
POST   /reindex
DELETE /documents/{filename}
DELETE /documents
```

### YouTube

```text
POST /youtube/transcript
```

### GitHub

```text
POST /github/index
POST /github/query
```

### Sessions

```text
/sessions
```

### Health

```text
/health
```

Interactive FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

---

## 🚀 Running NexaMind Locally

### 1. Clone the Repository

```bash
git clone https://github.com/annanyasinha/Nexamind.git
cd Nexamind
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root.

Example:

```env
GOOGLE_API_KEY=your_google_api_key

MONGODB_URI=your_mongodb_atlas_connection_string
MONGODB_DB_NAME=nexamind

GITHUB_TOKEN=your_optional_github_token

CHUNK_SIZE=1000
CHUNK_OVERLAP=200

MAX_FILE_SIZE_MB=25
```

`GITHUB_TOKEN` is optional for public repositories but can be configured for authenticated GitHub API access.

Never commit `.env` or API credentials to GitHub.

---

## ▶️ Start the FastAPI Backend

```bash
python3 app.py --backend
```

Backend:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 🖥️ Start the Streamlit Frontend

Open another terminal, activate the virtual environment, and run:

```bash
python3 app.py --frontend
```

Frontend:

```text
http://localhost:8501
```

---

## 💻 Direct CLI Query

NexaMind can also execute a direct document RAG query from the terminal:

```bash
python3 app.py --query "Explain the architecture of the uploaded document."
```

---

## 🐳 Running with Docker

Build and start the services:

```bash
docker compose up --build
```

Services:

```text
FastAPI Backend    → http://localhost:8000
Streamlit Frontend → http://localhost:8501
```

The project uses separate frontend and backend processes so the REST API and presentation layer can be independently managed.

---

## 🧪 Running Tests

Run the automated test suite with:

```bash
python3 -m pytest tests/ -v
```

The project includes tests for major application components such as document loading, session management, and API behavior.

---

## 🔐 Security & Reliability Features

NexaMind includes several safeguards:

```text
File extension validation
25 MB application-level upload size validation
Path traversal protection
SHA-256 content deduplication
Sensitive GitHub filename filtering
GitHub file-size filtering
API rate limiting
Environment-variable based secrets
External API retry handling
Persistent source metadata
```

---

## 🎯 Why NexaMind?

A basic RAG project often supports a single knowledge source such as PDFs.

NexaMind takes a **source-aware multi-source approach**.

Each source has a different strategy:

| Source | Ingestion | Storage | Freshness / Provenance |
|---|---|---|---|
| Documents | File parsing | Shared FAISS | SHA-256 checksum |
| YouTube | Transcript API | Per-video FAISS | Timestamp metadata |
| GitHub | GitHub REST API | Per-repository FAISS | Commit SHA + file provenance |
| Web | Live retrieval | Not permanently indexed | Current web sources |

The agent provides one interface across these different retrieval systems.

---

## ⚠️ Current Limitations

The current version has several known limitations:

- GitHub source-code chunking is primarily text/character based rather than AST-aware.
- A changed GitHub commit currently causes a repository-level index rebuild rather than updating only changed files.
- Local FAISS persistence depends on the deployment environment and persistent disk configuration.
- The current rate limiter uses in-memory state and is designed for a single backend instance.
- GitHub secret filtering is based mainly on file names, paths, extensions, and filtering rules rather than full secret scanning.
- Retrieval currently relies primarily on dense vector similarity rather than hybrid semantic + lexical search.
- Large-scale distributed background indexing is not implemented.

These limitations provide clear directions for future development.

---

## 🔮 Future Improvements

Potential future improvements include:

```text
AST / Tree-sitter based code-aware GitHub chunking
File-level incremental GitHub indexing using blob SHAs
Hybrid FAISS + BM25 retrieval
Cross-encoder or LLM reranking
Automated RAG evaluation pipeline
Shared persistent vector storage
Background indexing workers
RabbitMQ / task queues for long-running indexing
Redis for distributed caching and rate-limit state
Authentication and user-level authorization
Stronger secret detection
Prompt-injection defenses
Restricted production CORS configuration
Observability with structured metrics and tracing
```

These technologies would be introduced only when required by scale or application requirements rather than adding unnecessary architectural complexity.

---

## 📌 Engineering Highlights

NexaMind demonstrates:

- End-to-end RAG architecture
- Dense embedding generation
- Semantic vector retrieval
- FAISS index persistence
- Cosine similarity search
- Multi-source ingestion
- Source-aware provenance
- AI agent routing
- Incremental document indexing
- SHA-256 deduplication
- GitHub commit-based cache invalidation
- Persistent MongoDB chat sessions
- REST API design
- Pydantic validation
- SSE streaming
- API rate limiting
- Docker-based deployment

---

## 👩‍💻 Author

**Annanya Sinha**

Master of Computer Applications  
National Institute of Technology Raipur

GitHub: [annanyasinha](https://github.com/annanyasinha)

---

## 📄 License

This project was developed for educational, research, and portfolio purposes.