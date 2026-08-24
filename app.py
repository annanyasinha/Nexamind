#!/usr/bin/env python3

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path


# ============================================================
# PROJECT PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
src_path = BASE_DIR / "src"

# Add src directory to Python path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set PYTHONPATH for child processes
os.environ["PYTHONPATH"] = (
    str(src_path)
    + os.path.pathsep
    + os.environ.get("PYTHONPATH", "")
)


# ============================================================
# LOCAL VIRTUAL ENVIRONMENT SUPPORT
# ============================================================

venv_python = BASE_DIR / "venv" / "bin" / "python"

if (
    venv_python.exists()
    and os.environ.get("VIRTUAL_ENV") != str(venv_python.parent.parent)
    and sys.prefix == sys.base_prefix
):
    os.execv(
        str(venv_python),
        [str(venv_python)] + sys.argv
    )

if venv_python.exists():
    os.environ["PATH"] = (
        str(venv_python.parent)
        + os.path.pathsep
        + os.environ.get("PATH", "")
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import settings
from core.search_engine import RAGSearch
from utils.logger import logger


# ============================================================
# FASTAPI BACKEND
# ============================================================

def start_backend():
    """
    Start NexaMind FastAPI backend.

    Render automatically provides PORT.
    When running locally, settings.PORT is used.
    """

    import uvicorn

    host = "0.0.0.0"

    port = int(
        os.environ.get(
            "PORT",
            settings.PORT
        )
    )

    logger.info(
        f"Starting NexaMind FastAPI backend on "
        f"http://{host}:{port}"
    )

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False
    )


# ============================================================
# STREAMLIT FRONTEND
# ============================================================

def start_frontend():
    """
    Start NexaMind Streamlit frontend.

    Render provides PORT automatically.
    Locally, Streamlit uses 8501.
    """

    ui_script = src_path / "ui" / "app.py"

    if not ui_script.exists():
        logger.error(
            f"Streamlit application not found: {ui_script}"
        )
        sys.exit(1)

    port = int(
        os.environ.get(
            "PORT",
            "8501"
        )
    )

    logger.info(
        f"Starting NexaMind Streamlit frontend "
        f"on port {port}"
    )

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_script),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(port),
        "--server.headless",
        "true"
    ]

    subprocess.run(
        command,
        check=True
    )


# ============================================================
# RUN BACKEND + FRONTEND LOCALLY
# ============================================================

def start_all():
    """
    Start both backend and frontend.

    This mode is mainly intended for LOCAL DEVELOPMENT.

    For Render, use:
        APP_MODE=backend
    or
        APP_MODE=frontend
    """

    logger.info(
        "Starting NexaMind backend and frontend..."
    )

    backend_cmd = [
        sys.executable,
        __file__,
        "--backend"
    ]

    backend_proc = subprocess.Popen(
        backend_cmd
    )

    logger.info(
        "Backend started. Waiting for initialization..."
    )

    time.sleep(2)

    try:
        start_frontend()

    finally:

        logger.info(
            "Stopping NexaMind backend..."
        )

        backend_proc.terminate()

        try:
            backend_proc.wait(timeout=5)

        except subprocess.TimeoutExpired:
            backend_proc.kill()


# ============================================================
# CLI QUERY
# ============================================================

def run_query(query):
    """
    Run a NexaMind RAG query directly from the terminal.
    """

    logger.info(
        f"Executing query: {query}"
    )

    rag_search = RAGSearch()

    result = rag_search.search_with_sources(
        query,
        top_k=3
    )

    print("\n" + "=" * 60)

    print(
        f"QUERY: {result['query']}"
    )

    print("=" * 60)

    print("\n--- SUMMARY ---")

    print(
        result["summary"]
    )

    print(
        f"\n--- SOURCES ({len(result['sources'])}) ---"
    )

    for index, source in enumerate(
        result["sources"]
    ):

        print(
            f"[{index + 1}] "
            f"L2 Distance: {source['distance']:.4f} | "
            f"Content: {source['text'][:120]}..."
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NexaMind Application Launcher"
    )

    parser.add_argument(
        "--backend",
        "--api",
        action="store_true",
        help="Launch FastAPI backend"
    )

    parser.add_argument(
        "--frontend",
        "--ui",
        action="store_true",
        help="Launch Streamlit frontend"
    )

    parser.add_argument(
        "--all",
        "--both",
        action="store_true",
        help="Launch backend and frontend together"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Execute a RAG query"
    )

    args = parser.parse_args()


    # ========================================================
    # RENDER APP MODE
    # ========================================================

    app_mode = os.getenv(
        "APP_MODE",
        ""
    ).strip().lower()


    # Environment variable takes priority on Render

    if app_mode == "backend":

        logger.info(
            "APP_MODE=backend detected"
        )

        start_backend()
        return


    elif app_mode == "frontend":

        logger.info(
            "APP_MODE=frontend detected"
        )

        start_frontend()
        return


    # ========================================================
    # COMMAND LINE MODES
    # ========================================================

    if args.backend:

        start_backend()


    elif args.frontend:

        start_frontend()


    elif args.all:

        start_all()


    elif args.query:

        run_query(
            args.query
        )


    else:

        print("\n")
        print("=" * 60)
        print("NexaMind")
        print("=" * 60)

        print(
            "\nNo application mode specified."
        )

        print(
            "\nAvailable commands:"
        )

        print(
            "python app.py --backend"
        )

        print(
            "python app.py --frontend"
        )

        print(
            "python app.py --all"
        )

        print(
            'python app.py --query "your question"'
        )

        print(
            "\nFor Render:"
        )

        print(
            "APP_MODE=backend"
        )

        print(
            "or"
        )

        print(
            "APP_MODE=frontend"
        )

        print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
