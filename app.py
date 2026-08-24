def main():
    """Launches the RAG application components based on command line arguments."""

    parser = argparse.ArgumentParser(
        description="NexaMind Application Launcher"
    )

    parser.add_argument(
        "--backend", "--api",
        action="store_true",
        help="Launch FastAPI REST backend server"
    )

    parser.add_argument(
        "--frontend", "--ui",
        action="store_true",
        help="Launch Streamlit Web UI dashboard"
    )

    parser.add_argument(
        "--all", "--both",
        action="store_true",
        help="Launch both backend and frontend concurrently"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Execute a query directly via CLI"
    )

    args = parser.parse_args()

    if args.all:
        logger.info(
            "Starting both Backend REST server and Frontend UI dashboard..."
        )

        backend_cmd = [sys.executable, __file__, "--backend"]
        backend_proc = subprocess.Popen(backend_cmd)

        logger.info(
            "Backend process started. Waiting 2 seconds for server initialization..."
        )

        time.sleep(2)

        try:
            ui_script = src_path / "ui" / "app.py"

            subprocess.run([
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ui_script)
            ])

        finally:
            logger.info("Shutting down backend process...")
            backend_proc.terminate()

    elif args.backend:
        import uvicorn

        # Render provides PORT automatically.
        # Locally, settings.PORT is used as fallback.
        port = int(os.environ.get("PORT", settings.PORT))

        # Required for deployment.
        host = "0.0.0.0"

        logger.info(
            f"Starting FastAPI backend server on http://{host}:{port} ..."
        )

        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=False
        )

    elif args.frontend:
        logger.info("Launching Streamlit frontend dashboard ...")

        ui_script = src_path / "ui" / "app.py"

        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_script)
        ])

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
        print(f"QUERY: {res['query']}")
        print("=" * 50)

        print("\n--- SUMMARY ---")
        print(res["summary"])

        print(
            f"\n--- SOURCES ({len(res['sources'])}) ---"
        )

        for idx, src in enumerate(res["sources"]):
            print(
                f"[{idx+1}] "
                f"L2 Dist: {src['distance']:.4f} | "
                f"Content: {src['text'][:120]}..."
            )

    else:
        logger.info(
            "No flag specified. Running sample CLI query..."
        )

        rag_search = RAGSearch()

        query = "Where did Annanya study?"

        res = rag_search.search_with_sources(
            query,
            top_k=3
        )

        print("\n[Query]:", query)
        print("[Summary]:", res["summary"])

        print("\n" + "-" * 50)

        print(
            "💡 TIP: To open the interactive Web UI in your browser, run:"
        )

        print("   python3 app.py --frontend")
        print(
            "   or run both API backend & Web UI with: python3 app.py --all"
        )

        print("-" * 50)


if __name__ == "__main__":
    main()
