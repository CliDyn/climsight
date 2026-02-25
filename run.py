#!/usr/bin/env python3
"""
ClimSight Launcher — starts both the FastAPI backend and the React frontend dev server.

Usage:
    python run.py            # start both
    python run.py --backend  # backend only
    python run.py --frontend # frontend dev server only
"""

import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def start_backend(port: int = 8000):
    env = {**os.environ, "PYTHONPATH": ROOT}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(port), "--reload"],
        cwd=ROOT,
        env=env,
    )


def start_frontend():
    frontend_dir = os.path.join(ROOT, "frontend")
    if not os.path.isdir(os.path.join(frontend_dir, "node_modules")):
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Launch ClimSight")
    parser.add_argument("--backend", action="store_true", help="Start backend only")
    parser.add_argument("--frontend", action="store_true", help="Start frontend only")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    args = parser.parse_args()

    procs = []

    # If neither flag, start both
    both = not args.backend and not args.frontend

    try:
        if both or args.backend:
            print(f"🚀  Starting FastAPI backend on :{args.port}")
            procs.append(start_backend(args.port))
            time.sleep(1)

        if both or args.frontend:
            print("⚡  Starting React frontend dev server")
            procs.append(start_frontend())

        if both:
            print("\n✅  ClimSight is running!")
            print(f"   Backend:  http://localhost:{args.port}")
            print(f"   Frontend: http://localhost:5173")
            print("   Press Ctrl+C to stop.\n")

        # Wait for any process to exit
        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
                p.wait(timeout=5)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
