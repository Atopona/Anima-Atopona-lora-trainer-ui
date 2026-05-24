"""Run DiffSynth's training script while writing loss scalars to TensorBoard.

DiffSynth's runner passes `loss` to `ModelLogger.on_step_end(...)`, but the
default logger only saves checkpoints. This wrapper patches the logger before
executing the upstream training script, so we can keep using DiffSynth's own
training code without editing the checked-out DiffSynth-Studio tree.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diffsynth-train-script", required=True)
    parser.add_argument("--tensorboard-logdir", required=True)
    return parser.parse_known_args(argv)


def _loss_to_float(loss) -> float | None:
    if loss is None:
        return None
    try:
        if hasattr(loss, "detach"):
            loss = loss.detach()
        if hasattr(loss, "float"):
            loss = loss.float()
        if hasattr(loss, "mean"):
            loss = loss.mean()
        if hasattr(loss, "item"):
            loss = loss.item()
        return float(loss)
    except Exception:
        return None


def _patch_model_logger(log_dir: str):
    import diffsynth.diffusion as diffusion
    import diffsynth.diffusion.logger as logger_module

    base_logger = logger_module.ModelLogger

    class TensorBoardModelLogger(base_logger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tb_writer = None
            self._tb_log_dir = log_dir

        def _writer(self):
            if self._tb_writer is None:
                from torch.utils.tensorboard import SummaryWriter

                Path(self._tb_log_dir).mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(self._tb_log_dir)
            return self._tb_writer

        def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
            super().on_step_end(accelerator, model, save_steps, **kwargs)
            if not getattr(accelerator, "is_main_process", True):
                return
            value = _loss_to_float(kwargs.get("loss"))
            if value is None:
                return
            writer = self._writer()
            writer.add_scalar("loss", value, self.num_steps)
            writer.add_scalar("train/loss", value, self.num_steps)
            if self.num_steps % 10 == 0:
                writer.flush()

        def on_training_end(self, accelerator, model, save_steps=None):
            try:
                super().on_training_end(accelerator, model, save_steps)
            finally:
                if self._tb_writer is not None:
                    self._tb_writer.flush()
                    self._tb_writer.close()

    logger_module.ModelLogger = TensorBoardModelLogger
    diffusion.ModelLogger = TensorBoardModelLogger


def main() -> None:
    args, train_args = _parse_args(sys.argv[1:])
    train_script = str(Path(args.diffsynth_train_script).resolve())
    _patch_model_logger(args.tensorboard_logdir)
    sys.argv = [train_script, *train_args]
    runpy.run_path(train_script, run_name="__main__")


if __name__ == "__main__":
    main()
