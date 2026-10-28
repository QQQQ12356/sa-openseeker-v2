"""Serve a trained Plain checkpoint with the upstream SGLang completion interface."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from search_comp.workflows.benchmark_openseeker import resolve_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30018)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.port < 65536 or min(args.tp_size, args.context_length) < 1:
        parser.error("invalid port, tensor parallel size or context length")
    model, kind = resolve_checkpoint(args.run_dir, args.model, None)
    config = Path(model) / "config.json"
    config_data = json.loads(config.read_text()) if config.exists() else {}
    if kind == "beacon" or (Path(model) / "beacon_config.json").exists() or config_data.get("beacon") or config_data.get("enable_beacon"):
        parser.error("Beacon needs --mode native in 47; standard SGLang does not execute Beacon compression")
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", model,
               "--host", args.host, "--port", str(args.port), "--tp-size", str(args.tp_size),
               "--context-length", str(args.context_length)]
    print("[serve] " + shlex.join(command), flush=True)
    print(f"[serve] OPENSEEKER_BASE_URL=http://{args.host}:{args.port}/v1/completions", flush=True)
    print(f"[serve] OPENSEEKER_MODEL={model}", flush=True)
    if not args.dry_run:
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
