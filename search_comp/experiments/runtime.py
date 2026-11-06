"""Artifact ownership, provenance and failure records for every public script."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]


def path(value):
    value = Path(value).expanduser()
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def slug(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError(f"名称只能使用字母、数字、下划线、连字符（1–96 字符）: {value!r}")
    return value


def stamp(label):
    return f"{slug(label)}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:6]}"


def digest(filename):
    result = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(filename, value):
    filename = Path(filename)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(filename)


def read_json(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def configure(environment):
    if platform.system() != "Linux":
        raise RuntimeError("本项目运行入口仅支持 Linux；Windows 仅修改代码和运行静态/单元检查。")
    os.chdir(ROOT)
    os.environ.update({key: str(value) for key, value in environment.items() if value is not None})
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/cache/huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts/cache"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / "artifacts/cache/torch"))
    os.environ.setdefault("WANDB_MODE", "offline")


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


@contextmanager
def job(directory, operation, settings):
    """Reserve once; archive the edited script, hashes, logs and terminal state."""
    directory = path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    script = Path(sys.argv[0]).resolve()
    if script.is_file():
        shutil.copyfile(script, directory / "source_script.py")
    launcher = os.environ.get("SA_LAUNCHER")
    if launcher and Path(launcher).is_file():
        shutil.copyfile(launcher, directory / "source_launcher.sh")
    metadata = dict(operation=operation, settings=settings, status="running",
                    created_at=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    platform=platform.platform(),
                    code_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "search_comp").rglob("*.py"))},
                    environment={k: os.environ.get(k) for k in ("HF_ENDPOINT", "HF_HOME", "CUDA_VISIBLE_DEVICES",
                                 "NPROC_PER_NODE", "DISTRIBUTED_BACKEND", "HF_HUB_OFFLINE", "WANDB_MODE")})
    write_json(directory / "job.json", metadata)
    with (directory / "console.log").open("x", encoding="utf-8") as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            try:
                yield directory
                metadata["status"] = "complete"
            except BaseException as error:
                metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
                traceback.print_exc()
                raise
            finally:
                metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
                write_json(directory / "job.json", metadata)
    print(f"产物目录: {directory}", flush=True)


def execute(command):
    """Use a subprocess only for distributed training / model-memory isolation."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
