#!/usr/bin/env python3
"""Persistent model server: loads all models once, serves requests via Unix socket.

Usage:
    # Start server (loads SAM3, GroundingDINO, etc.)
    python scripts/model_server.py start

    # Stop server
    python scripts/model_server.py stop

    # Check status
    python scripts/model_server.py status
"""

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SOCKET_PATH = "/tmp/food_agent_model_server.sock"
PID_FILE = "/tmp/food_agent_model_server.pid"


def start_server():
    """Start the model server."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            old_pid = int(f.read().strip())
        try:
            os.kill(old_pid, 0)
            print(f"Server already running (PID {old_pid})")
            return
        except OSError:
            os.remove(PID_FILE)
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    print("Loading pipeline (light mode, no heavy models)...")
    t0 = time.time()

    from food_agent.agent_v2.pipeline import Pipeline
    pipeline = Pipeline(load_models=False)
    pipeline.agent.max_iterations = 8
    pipeline.agent.timeout = 120

    print(f"Pipeline loaded in {time.time()-t0:.1f}s")
    print(f"  Recipes: {len(pipeline.recipe_kb._recipes) if hasattr(pipeline.recipe_kb, '_recipes') else 0}")

    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Start socket server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(16)  # Allow more pending connections
    server.settimeout(1.0)

    print(f"Server listening on {SOCKET_PATH} (max_workers=4)")

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def handle_request(conn):
        """Handle a single request in a thread."""
        try:
            data = conn.recv(1024 * 1024 * 10)
            if not data:
                conn.close()
                return

            request = json.loads(data.decode())
            action = request.get("action", "")

            if action == "answer":
                t1 = time.time()
                result = pipeline.answer(
                    question=request["question"],
                    video_id=request.get("video_id", ""),
                    choices=request.get("choices"),
                )
                result["latency"] = round(time.time() - t1, 1)
                conn.sendall(json.dumps(result, default=str).encode())

            elif action == "status":
                conn.sendall(json.dumps({
                    "status": "running",
                    "pid": os.getpid(),
                    "sam3": pipeline.sam3 is not None,
                    "gdino": pipeline.gdino is not None,
                    "recipes": len(pipeline.recipe_kb._recipes) if hasattr(pipeline.recipe_kb, '_recipes') else 0,
                }).encode())

            elif action == "shutdown":
                conn.sendall(json.dumps({"status": "shutting_down"}).encode())
                nonlocal running
                running = False

            else:
                conn.sendall(json.dumps({"error": f"unknown action: {action}"}).encode())

        except Exception as e:
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode())
            except Exception:
                pass
        finally:
            conn.close()

    # Thread pool for handling concurrent requests
    executor = ThreadPoolExecutor(max_workers=4)

    while running:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        except Exception:
            break

        executor.submit(handle_request, conn)

    executor.shutdown(wait=False)
    server.close()

    # Cleanup
    server.close()
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    print("Server stopped")


def stop_server():
    """Stop the model server."""
    if not os.path.exists(PID_FILE):
        print("Server not running")
        return

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    try:
        # Send shutdown via socket
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCKET_PATH)
        client.sendall(json.dumps({"action": "shutdown"}).encode())
        client.recv(4096)
        client.close()
    except Exception:
        pass

    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    except OSError:
        pass

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    print("Server stopped")


def server_status():
    """Check server status."""
    if not os.path.exists(PID_FILE):
        print("Server not running")
        return False

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, 0)
    except OSError:
        print(f"Stale PID file (process {pid} not found)")
        return False

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCKET_PATH)
        client.sendall(json.dumps({"action": "status"}).encode())
        resp = json.loads(client.recv(4096).decode())
        client.close()
        print(f"Server running (PID {pid})")
        print(f"  SAM3: {resp.get('sam3', False)}")
        print(f"  GroundingDINO: {resp.get('gdino', False)}")
        print(f"  Recipes: {resp.get('recipes', 0)}")
        return True
    except Exception as e:
        print(f"Server not responding: {e}")
        return False


def send_request(request: dict) -> dict:
    """Send a request to the model server."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(300)
    client.connect(SOCKET_PATH)
    client.sendall(json.dumps(request, default=str).encode())

    chunks = []
    while True:
        chunk = client.recv(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    client.close()

    return json.loads(b"".join(chunks).decode())


def main():
    parser = argparse.ArgumentParser(description="Model Server")
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()

    if args.action == "start":
        start_server()
    elif args.action == "stop":
        stop_server()
    elif args.action == "status":
        server_status()


if __name__ == "__main__":
    main()
