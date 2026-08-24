#!/usr/bin/env python3

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Virtual environment (for local development)
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


# Add src directory to Python path
src_path = BASE_DIR / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Set PYTHONPATH
os.environ["PYTHONPATH"] = (
    str(src_path)
    + os.path.pathsep
    + os.environ.get("PYTHONPATH", "")
)


# Add virtual environment to PATH locally
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
# FASTAPI SERVER
# ============================================================

def start_backend():
    """
    Start the FastAPI backend.

    Render automatically provides the PORT environment variable.
    Locally, settings.PORT is used as the fallback.
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
        f"Starting FastAPI backend on http://{host}:{port}"
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

def start_frontend(port=None):
    """
    Start the Streamlit frontend.
    """

    ui_script = src_path / "ui" / "app.py"

    if not ui_script.exists():
        logger.error(
            f"Streamlit UI file not found: {ui_script}"
        )
        return

    if port is None:
        port = 8501

    logger.info(
        f"Starting Streamlit frontend on port {port}"
    )

    subprocess.run(
        [
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
    )


# ============================================================
# RUN BACKEND + FRONTEND
# ============================================================

def start_all():
    """
    Start FastAPI and Streamlit together.

    FastAPI uses Render's public PORT.

    Streamlit runs internally on port 8501.
    """

    logger.info(
        "Starting NexaMind FastAPI backend + Streamlit frontend..."
    )

    # Start Streamlit in background
    ui_script = src_path / "ui" / "app.py"

    if not ui_script.exists():
        logger.error(
            f"Streamlit UI file not found: {ui_script}"
        )
        return

    frontend_port = 8501

    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_script),

        "--server.address",
        "0.0.0.0",

        "--server.port",
        str(frontend_port),

        "--server.headless",
        "true"
    ]

    logger.info(
        f"Starting Streamlit internally on port {frontend_port}"
    )

    frontend_proc = subprocess.Popen(frontend_cmd)

    # Give Streamlit time to initialize
    time.sleep(2)

    try:

        # FastAPI stays as the main process
        start_backend()

    finally:

        logger.info(
            "Stopping Streamlit frontend..."
        )

        frontend_proc.terminate()

        try:
            frontend_proc.wait(timeout=5)

        except subprocess.TimeoutExpired:
            frontend_proc.kill()


# ============================================================
# CLI QUERY
# ============================================================

def run_query(query):
    """
    Execute a RAG query directly from the command line.
    """

    logger.info(
        f"Executing CLI RAG query: '{query}'"
    )

    rag_search = RAGSearch()

    res = rag_search.search_with_sources(
        query,
        top_k=3
    )

    print("\n" + "=" * 50)

    print(
        f"QUERY: {res['query']}"
    )

    print("=" * 50)

    print("\n--- SUMMARY ---")

    print(
        res["summary"]
    )

    print(
        f"\n--- SOURCES ({len(res['sources'])}) ---"
    )

    for idx, src in enumerate(
        res["sources"]
    ):

        print(
            f"[{idx + 1}] "
            f"L2 Dist: {src['distance']:.4f} | "
            f"Content: {src['text'][:120]}..."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NexaMind Application Launcher"
    )

    parser.add_argument(
        "--backend",
        "--api",
        action="store_true",
        help="Launch FastAPI REST backend"
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
        help="Launch FastAPI and Streamlit"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Execute a RAG query from CLI"
    )

    args = parser.parse_args()


    # --------------------------------------------------------
    # BOTH
    # --------------------------------------------------------

    if args.all:

        start_all()


    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    elif args.backend:

        start_backend()


    # --------------------------------------------------------
    # FRONTEND
    # --------------------------------------------------------

    elif args.frontend:

        # Render provides PORT
        port = int(
            os.environ.get(
                "PORT",
                "8501"
            )
        )

        start_frontend(port)


    # --------------------------------------------------------
    # CLI QUERY
    # --------------------------------------------------------

    elif args.query:

        run_query(
            args.query
        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    else:

        print("\nNexaMind")
        print("=" * 50)

        print(
            "No command specified."
        )

        print("\nAvailable commands:")

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

        print("=" * 50)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
