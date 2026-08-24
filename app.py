#!/usr/bin/env python3

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path


# ============================================================
# PROJECT / VIRTUAL ENVIRONMENT SETUP
# ============================================================

# Automatically switch to project virtual environment
# when running locally, if one exists.
venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"

if (
    venv_python.exists()
    and os.environ.get("VIRTUAL_ENV") != str(venv_python.parent.parent)
    and sys.prefix == sys.base_prefix
):
    os.execv(
        str(venv_python),
        [str(venv_python)] + sys.argv
    )


# ============================================================
# PYTHON PATH SETUP
# ============================================================

# Add src directory to Python path so imports such as
# "from config import settings" work correctly.
src_path = Path(__file__).resolve().parent / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Set PYTHONPATH for child processes.
os.environ["PYTHONPATH"] = (
    str(src_path)
    + os.path.pathsep
    + os.environ.get("PYTHONPATH", "")
)


# Add virtual environment binaries to PATH when available.
if venv_python.exists():
    os.environ["PATH"] = (
        str(venv_python.parent)
        + os.path.pathsep
        + os.environ.get("PATH", "")
    )


# ============================================================
# NEXAMIND IMPORTS
# ============================================================

from config import settings
from core.search_engine import RAGSearch
from utils.logger import logger


# ============================================================
# APPLICATION LAUNCHER
# ============================================================

def main():
    """
    Launch NexaMind components based on command-line arguments.

    Supported modes:

    --backend / --api
        Start FastAPI backend.

    --frontend / --ui
        Start Streamlit frontend.

    --all / --both
        Start backend and frontend together.

    --query
        Run a RAG query directly from the command line.
    """

    parser = argparse.ArgumentParser(
        description="NexaMind Application Launcher"
    )

    parser.add_argument(
        "--backend",
        "--api",
        action="store_true",
        help="Launch FastAPI REST backend server"
    )

    parser.add_argument(
        "--frontend",
        "--ui",
        action="store_true",
        help="Launch Streamlit Web UI dashboard"
    )

    parser.add_argument(
        "--all",
        "--both",
        action="store_true",
        help="Launch both backend and frontend concurrently"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Execute a query directly via CLI"
    )

    args = parser.parse_args()


    # ========================================================
    # RUN BACKEND + FRONTEND
    # ========================================================

    if args.all:

        logger.info(
            "Starting both Backend REST server "
            "and Frontend UI dashboard..."
        )

        # Start backend as a separate process.
        backend_cmd = [
            sys.executable,
            __file__,
            "--backend"
        ]

        backend_proc = subprocess.Popen(backend_cmd)

        logger.info(
            "Backend process started. "
            "Waiting 2 seconds for server initialization..."
        )

        time.sleep(2)

        try:

            # Start Streamlit frontend.
            ui_script = src_path / "ui" / "app.py"

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ui_script)
                ]
            )

        finally:

            logger.info(
                "Shutting down backend process..."
            )

            backend_proc.terminate()


    # ========================================================
    # RUN FASTAPI BACKEND
    # ========================================================

    elif args.backend:

        import uvicorn

        # Render automatically provides a PORT environment
        # variable.
        #
        # On Render:
        #     PORT -> Render supplied port
        #
        # Locally:
        #     Falls back to settings.PORT (normally 8000)
        port = int(
            os.environ.get(
                "PORT",
                settings.PORT
            )
        )

        # 0.0.0.0 is required so Render can access
        # the application from outside the container.
        host = "0.0.0.0"

        logger.info(
            f"Starting FastAPI backend server "
            f"on http://{host}:{port} ..."
        )

        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,

            # Do NOT enable reload in production.
            reload=False
        )


    # ========================================================
    # RUN STREAMLIT FRONTEND
    # ========================================================

    elif args.frontend:

        logger.info(
            "Launching Streamlit frontend dashboard..."
        )

        ui_script = (
            src_path
            / "ui"
            / "app.py"
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ui_script)
            ]
        )


    # ========================================================
    # RUN CLI QUERY
    # ========================================================

    elif args.query:

        logger.info(
            f"Executing CLI RAG query: '{args.query}'"
        )

        rag_search = RAGSearch()

        res = rag_search.search_with_sources(
            args.query,
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
            f"\n--- SOURCES "
            f"({len(res['sources'])}) ---"
        )

        for idx, src in enumerate(
            res["sources"]
        ):

            print(
                f"[{idx + 1}] "
                f"L2 Dist: "
                f"{src['distance']:.4f} | "
                f"Content: "
                f"{src['text'][:120]}..."
            )


    # ========================================================
    # DEFAULT MODE
    # ========================================================

    else:

        logger.info(
            "No flag specified. "
            "Running sample CLI query..."
        )

        rag_search = RAGSearch()

        query = "Where did Annanya study?"

        res = rag_search.search_with_sources(
            query,
            top_k=3
        )

        print(
            "\n[Query]:",
            query
        )

        print(
            "[Summary]:",
            res["summary"]
        )

        print(
            "\n" + "-" * 50
        )

        print(
            "💡 TIP: To open the interactive "
            "Web UI in your browser, run:"
        )

        print(
            "   python3 app.py --frontend"
        )

        print(
            "   or run both API backend & "
            "Web UI with:"
        )

        print(
            "   python3 app.py --all"
        )

        print(
            "-" * 50
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
