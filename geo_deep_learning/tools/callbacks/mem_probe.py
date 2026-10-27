"""GPU memory probe for the first training steps."""

import logging
import os
import sys
from pathlib import Path

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)

_GIB = 2**30


class MemProbe(Callback):
    """Log CUDA memory at batch start and end, and dump a snapshot."""

    def __init__(
        self,
        steps: int = 30,
        snapshot_dir: str | None = None,
    ) -> None:
        """Record the first `steps` batches on rank 0."""
        self.steps = steps
        self.snapshot_dir = snapshot_dir
        self._recording = False

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Start the CUDA allocation history on rank 0."""
        if not trainer.is_global_zero or not torch.cuda.is_available():
            return
        torch.cuda.memory._record_memory_history(max_entries=100_000)  # noqa: SLF001
        self._recording = True

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        batch: object,
        batch_idx: int,
    ) -> None:
        """Log memory before the forward. This still runs if the step OOMs."""
        if not self._active(trainer, batch_idx) or not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        self._log("start", batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: object,  # noqa: ARG002
        batch: object,
        batch_idx: int,
    ) -> None:
        """Log memory after the step, then stop recording once enough steps ran."""
        if not self._active(trainer, batch_idx) or not torch.cuda.is_available():
            return
        self._log("end", batch, batch_idx)
        if batch_idx + 1 >= self.steps:
            self._dump()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        exception: BaseException,  # noqa: ARG002
    ) -> None:
        """Dump the snapshot when the run dies before `steps` completes."""
        if trainer.is_global_zero:
            self._dump()

    def _active(self, trainer: Trainer, batch_idx: int) -> bool:
        return trainer.is_global_zero and batch_idx < self.steps

    def _log(self, tag: str, batch: object, batch_idx: int) -> None:
        image = wave = None
        if isinstance(batch, dict):
            image = batch.get("image")
            wave = batch.get("wavelengths")
        image_shape = tuple(image.shape) if torch.is_tensor(image) else None
        wave_shape = tuple(wave.shape) if torch.is_tensor(wave) else None
        alloc = torch.cuda.memory_allocated() / _GIB
        peak = torch.cuda.max_memory_allocated() / _GIB
        logger.info(
            "mem %s %d: alloc=%.2f GiB peak=%.2f GiB image=%s wavelengths=%s",
            tag,
            batch_idx,
            alloc,
            peak,
            image_shape,
            wave_shape,
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def _dump(self) -> None:
        if not self._recording:
            return
        self._recording = False
        job = os.environ.get("SLURM_JOB_ID", "local")
        directory = Path(self.snapshot_dir or ".")
        path = directory / f"dinov3_mem_{job}.pickle"
        directory.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))  # noqa: SLF001
        torch.cuda.memory._record_memory_history(enabled=None)  # noqa: SLF001
        logger.info("mem snapshot: %s", path)
        sys.stdout.flush()
        sys.stderr.flush()
