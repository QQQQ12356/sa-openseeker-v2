"""标准 Transformers Trainer 的日志与产物回调。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .runtime import append_jsonl, write_json


class JsonlMetricsCallback(TrainerCallback):
    """将 Trainer 日志实时追加到 ``trainer_metrics.jsonl``。"""

    def __init__(self, run_dir: str | Path, append: bool = False):
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "trainer_metrics.jsonl"
        self.started_at = time.time()
        self.append = append

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.metrics_path.exists() and not self.append:
            self.metrics_path.unlink()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if not logs:
            return
        record: dict[str, Any] = {
            "event": "log",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "elapsed_seconds": round(time.time() - self.started_at, 3),
        }
        record.update(logs)
        append_jsonl(self.metrics_path, record)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        write_json(
            self.run_dir / "trainer_state_summary.json",
            {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
                "elapsed_seconds": round(time.time() - self.started_at, 3),
                "log_history": state.log_history,
            },
        )


class CudaMemoryCallback(TrainerCallback):
    """Record allocator peaks per rank after each optimizer step and at train end."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _record(self, state, event):
        import os
        import torch
        if not torch.cuda.is_available():
            return
        rank = int(os.environ.get('RANK', '0'))
        append_jsonl(self.run_dir / f'cuda_memory_rank{rank}.jsonl', {
            'event': event, 'global_step': state.global_step, 'rank': rank,
            'device': torch.cuda.current_device(),
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        })

    def on_step_end(self, args, state, control, **kwargs):
        self._record(state, 'step_end')

    def on_train_end(self, args, state, control, **kwargs):
        self._record(state, 'train_end')
