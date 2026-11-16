"""Report per-question completion from in-process or worker predictions."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time


def generation_label(benchmark):
    return f"benchmark {benchmark.lower()} generation"


class CompletionProgress:
    """Count unique, fully written records; incomplete lines wait for the next poll."""

    def __init__(self, paths, total, label, key="id", item_type=None, emit=None):
        self.paths = [Path(path) for path in paths]
        self.total, self.label = total, label
        self.key, self.item_type = key, item_type
        self.emit = emit or (lambda line: print(line, flush=True))
        self.offsets = [0] * len(self.paths)
        self.buffers = [b""] * len(self.paths)
        self.seen = set()
        self.started = time.monotonic()
        self._report()

    def _report(self, row=None):
        fields = [f"[{self.label}] done={len(self.seen)}/{self.total}"]
        if row is not None:
            if row.get("status"):
                fields.append(f"status={row['status']}")
            seconds = row.get("seconds", row.get("elapsed_seconds"))
            if isinstance(seconds, (int, float)):
                fields.append(f"sample={seconds:.1f}s")
        fields.append(f"elapsed={time.monotonic() - self.started:.1f}s")
        self.emit(" ".join(fields))

    def record(self, row):
        """Count a prediction after its complete line has been flushed."""
        if not isinstance(row, dict) or (self.item_type is not None and row.get("type") != self.item_type):
            return
        identity = row.get(self.key)
        if not isinstance(identity, (str, int)) or identity in self.seen:
            return
        self.seen.add(identity)
        self._report(row)

    def poll(self):
        for index, path in enumerate(self.paths):
            try:
                with path.open("rb") as source:
                    source.seek(self.offsets[index])
                    data = source.read()
                    self.offsets[index] = source.tell()
            except FileNotFoundError:
                continue
            lines = (self.buffers[index] + data).split(b"\n")
            self.buffers[index] = lines.pop()
            for line in lines:
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue  # Scoring/merge validation owns malformed output errors.
                self.record(row)

    @contextmanager
    def watch(self):
        """Keep reporting even when a subprocess produces no stdout."""
        stopped = threading.Event()

        def monitor():
            while not stopped.wait(0.1):
                self.poll()

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            yield self
        finally:
            stopped.set()
            thread.join()
            self.poll()
