#!/usr/bin/env python3
import sys
import os
import time
import subprocess
import argparse
from pathlib import Path

# Auto-switch to project virtual environment if available and executed outside venv
venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
if venv_python.exists() and os.environ.get("VIRTUAL_ENV") != str(venv_python.parent.parent) and sys.prefix == sys.base_prefix:
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Add src directory to python path for seamless executions
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set PYTHONPATH environment variable for child processes (e.g. Streamlit runner)
os.environ["PYTHONPATH"] = str(src_path) + os.path.pathsep + os.environ.get("PYTHONPATH", "")

from config import settings
from core.search_engine import RAGSearch
from utils.logger import logger


def main():
    """Launches the RAG application components based on command line arguments."""
    parser = argparse.ArgumentParser(description="NexaMind Application Launcher")
    parser.add_argument("--backend", "--api", action="store_true", help="Launch FastAPI REST backend server")
    parser.add_argument("--frontend", "--ui", action="store_true", help="Launch Streamlit Web UI dashboard")
    parser.add_argument("--all", "--both", action="store_true", help="Launch both backend and frontend concurrently")
    parser.add_argument("--query", type=str, help="Execute a query directly via CLI")

    args = parser.parse_args()

    if args.all:
        logger.info("Starting both Backend REST server and Frontend UI dashboard...")
        # Start backend in a subprocess
        backend_cmd = [sys.executable, __file__, "--backend"]
        backend_proc = subprocess.Popen(backend_cmd)
        logger.info("Backend process started. Waiting 2 seconds for server initialization...")
        time.sleep(2)
        
        # Start frontend UI
        try:
            ui_script = src_path / "ui" / "app.py"
            os.system(f"streamlit run {ui_script}")
        finally:
            logger.info("Shutting down backend process...")
            backend_proc.terminate()

    elif args.backend:
        logger.info(f"Starting FastAPI backend server on http://{settings.HOST}:{settings.PORT} ...")
        import uvicorn
        uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
    elif args.frontend:
        logger.info("Launching Streamlit frontend dashboard ...")
        ui_script = src_path / "ui" / "app.py"
        os.system(f"streamlit run {ui_script}")

    elif args.query:
        logger.info(f"Executing CLI RAG query: '{args.query}'")
        rag_search = RAGSearch()
        res = rag_search.search_with_sources(args.query, top_k=3)
        print("\n" + "=" * 50)
        print(f"QUERY: {res['query']}")
        print("=" * 50)
        print("\n--- SUMMARY ---")
        print(res["summary"])
        print(f"\n--- SOURCES ({len(res['sources'])}) ---")
        for idx, src in enumerate(res["sources"]):
            print(f"[{idx+1}] L2 Dist: {src['distance']:.4f} | Content: {src['text'][:120]}...")
    else:
        logger.info("No flag specified. Running sample CLI query...")
        rag_search = RAGSearch()
        query = "Where did shubham study?"
        res = rag_search.search_with_sources(query, top_k=3)
        print("\n[Query]:", query)
        print("[Summary]:", res["summary"])
        print("\n" + "-" * 50)
        print("💡 TIP: To open the interactive Web UI in your browser, run:")
        print("   python3 app.py --frontend")
        print("   or run both API backend & Web UI with: python3 app.py --all")
        print("-" * 50)


if __name__ == "__main__":
    main()
