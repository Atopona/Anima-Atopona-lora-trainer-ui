"""
Anima LoRA Trainer — Local Gradio UI
Supports kohya-ss/sd-scripts and DiffSynth-Studio backends,
with TensorBoard logging and Chinese / English UI.
"""

import csv
import json
import math
import os
import re
import shutil
import shlex
import socket
import subprocess
import sys
import threading
import time
import atexit
import importlib.metadata as importlib_metadata
from datetime import datetime
from pathlib import Path

import gradio as gr
import toml

from i18n import t, get_lang, set_lang, SUPPORTED_LANGS

try:
    from pyngrok import ngrok as _ngrok
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False
    _ngrok = None

IS_COLAB = ("COLAB_GPU" in os.environ) or ("COLAB_RELEASE_TAG" in os.environ)

# ---------------------------------------------------------------------------
# Paths (all relative to the project root where app.py lives)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
CONFIGS_DIR = ROOT / "configs"
LOGS_DIR = ROOT / "logs"
TB_LOGS_ROOT = LOGS_DIR / "tb"
MODELS_DIR = ROOT / "models" / "anima"
SD_SCRIPTS_DIR = ROOT / "sd-scripts"
DIFFSYNTH_DEFAULT_DIR = ROOT / "DiffSynth-Studio"
DIFFSYNTH_GIT_URL = "https://github.com/modelscope/DiffSynth-Studio.git"
DIFFSYNTH_LEGACY_ANIMA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
TORCHAO_MIN_EXCLUSIVE_VERSION = (0, 16, 0)
TORCHAO_PIP_SPEC = "torchao>0.16.0"

DIT_MODEL = MODELS_DIR / "dit" / "anima-preview.safetensors"
QWEN3_MODEL = MODELS_DIR / "text_encoder" / "qwen_3_06b_base.safetensors"
VAE_MODEL = MODELS_DIR / "vae" / "qwen_image_vae.safetensors"
TRAIN_SCRIPT = SD_SCRIPTS_DIR / "anima_train_network.py"
DIFFSYNTH_TRAIN_SCRIPT_REL = "examples/anima/model_training/train.py"

BASE_MODEL_URLS = {
    "anima-base-v1.0": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
    "anima-preview3-base": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview3-base.safetensors",
    "anima-preview": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview.safetensors",
}


def get_dit_model_path(base_model: str) -> Path:
    filenames = {
        "anima-base-v1.0": "anima-base-v1.0.safetensors",
        "anima-preview": "anima-preview.safetensors",
        "anima-preview3-base": "anima-preview3-base.safetensors",
    }
    return MODELS_DIR / "dit" / filenames.get(base_model, "anima-base-v1.0.safetensors")


CONFIGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TB_LOGS_ROOT.mkdir(exist_ok=True, parents=True)

# Project-local accelerate config — keeps use_cpu=false and mixed_precision=bf16
# scoped to this app only. See app_configs/accelerate_gpu.yaml to change these.
# Absolute path so cwd switching (DiffSynth backend uses ds_dir as cwd) doesn't break resolution.
ACCELERATE_CONFIG = str(ROOT / "app_configs" / "accelerate_gpu.yaml")


def resolve_accelerate_launch_cmd() -> list[str]:
    """Return a working Accelerate launcher prefix."""
    candidates: list[list[str]] = []
    exe_dir = Path(sys.executable).resolve().parent

    if os.name == "nt":
        candidates.extend([
            [str(exe_dir / "Scripts" / "accelerate.exe"), "launch"],
            [str(exe_dir / "accelerate.exe"), "launch"],
            [str(exe_dir / "Scripts" / "accelerate-launch.exe")],
            [str(exe_dir / "accelerate-launch.exe")],
        ])
    else:
        candidates.extend([
            [str(exe_dir / "accelerate"), "launch"],
            [str(exe_dir / "accelerate-launch")],
        ])

    path_accelerate = shutil.which("accelerate")
    if path_accelerate:
        candidates.append([path_accelerate, "launch"])

    path_accelerate_launch = shutil.which("accelerate-launch")
    if path_accelerate_launch:
        candidates.append([path_accelerate_launch])

    seen = set()
    for cmd in candidates:
        executable = cmd[0]
        if executable in seen:
            continue
        seen.add(executable)
        if Path(executable).exists():
            return cmd

    return [sys.executable, "-m", "accelerate.commands.launch"]

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Language / backend (new)
    "language": "en",
    "backend": "kohya",
    "diffsynth_dir": "",
    # TensorBoard (new)
    "use_tensorboard": True,
    "tb_port": 6006,
    "tb_logdir": "",  # auto-derived per run if empty
    "ngrok_enable": False,
    "ngrok_token": "",
    # DiffSynth-specific (new)
    "lora_target_modules": "",
    "dataset_repeat": 50,
    "max_pixels": 1048576,
    "save_steps_ds": 0,
    # Basic
    "project_name": "my_lora",
    "base_model": "anima-base-v1.0",
    "image_directory": "",
    "output_directory": "",
    "network_dim": 20,
    "network_alpha": 20,
    "learning_rate": 0.0001,
    "max_train_epochs": 10,
    "resolution": 768,
    "repeats": 10,
    "caption_dropout": 0.1,
    "gpu_index": "0",
    # Advanced
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_num_cycles": 1,
    "lr_warmup_steps": 100,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_every_n_epochs": 1,
    "save_last_n_epochs": 4,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "seed": 42,
    "noise_offset": 0.03,
    "multires_noise_discount": 0.3,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "num_cpu_threads_per_process": 1,
    # Internal
    "last_train_config": "",
    "last_dataset_config": "",
    "last_diffsynth_args": "",
    "last_tb_logdir": "",
}


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except Exception:
            pass
    cfg["lora_target_modules"] = normalize_diffsynth_lora_target_modules(
        cfg.get("lora_target_modules", "")
    )
    return cfg


def save_config(cfg: dict):
    # Preserve any keys not in DEFAULTS (e.g. language was already saved by i18n)
    existing = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(cfg)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return ["CPU (no CUDA detected)"]
        choices = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            choices.append(f"{i}: {name}")
        return choices if choices else ["0", "1"]
    except ImportError:
        return ["0", "1"]


GPU_CHOICES = detect_gpus()


def gpu_index_from_choice(choice: str) -> str:
    if not choice:
        return "0"
    return str(choice).split(":")[0].strip()


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_dataset(image_dir: str) -> tuple[int, list[str], list[str]]:
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    all_files = list(p.iterdir())
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
    txt_basenames = {f.stem for f in all_files if f.suffix.lower() == ".txt" and f.is_file()}

    missing = [f.name for f in image_files if f.stem not in txt_basenames]
    warnings = []
    if not image_files:
        warnings.append("No image files found in directory.")
    if missing:
        warnings.append(f"{len(missing)} image(s) are missing caption (.txt) files.")
    return len(image_files), missing, warnings


# ---------------------------------------------------------------------------
# DiffSynth metadata.csv generation
# ---------------------------------------------------------------------------

def generate_diffsynth_metadata(image_dir: str, output_path: Path) -> tuple[Path, int]:
    """Scan a kohya-style flat dir and write Anima DiffSynth metadata.csv."""
    rows = []
    for img in sorted(Path(image_dir).iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        txt = img.with_suffix(".txt")
        caption = ""
        if txt.exists():
            try:
                caption = txt.read_text(encoding="utf-8").strip().replace("\n", " ")
            except Exception:
                caption = ""
        rows.append({"image": img.name, "prompt": caption})
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path, len(rows)


# ---------------------------------------------------------------------------
# kohya TOML config generation (ported directly from the notebook)
# ---------------------------------------------------------------------------

def create_kohya_training_config(
    project_name, output_dir, dit_model_path, qwen3_model_path, vae_model_path,
    network_dim=20, network_alpha=20, learning_rate=1e-4, max_train_epochs=10,
    optimizer_type="AdamW8bit", lr_scheduler="cosine_with_restarts",
    lr_scheduler_num_cycles=1, lr_warmup_steps=100,
    train_batch_size=1, gradient_accumulation_steps=1, max_grad_norm=1.0,
    save_every_n_epochs=1, save_last_n_epochs=4,
    mixed_precision="bf16", gradient_checkpointing=True,
    seed=42, noise_offset=0.03, multires_noise_discount=0.3,
    timestep_sampling="sigmoid", discrete_flow_shift=1.0,
    cache_latents=True, cache_text_encoder_outputs=True,
    vae_chunk_size=64, vae_disable_cache=True,
    logging_dir: str = "",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_training_{current_date}.toml"

    training_config = {
        "pretrained_model_name_or_path": str(dit_model_path),
        "qwen3": str(qwen3_model_path),
        "vae": str(vae_model_path),
        "network_module": "networks.lora_anima",
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "network_train_unet_only": True,
        "learning_rate": float(learning_rate),
        "optimizer_type": optimizer_type,
        "optimizer_args": ["weight_decay=0.1", "betas=[0.9, 0.99]"],
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "max_train_epochs": int(max_train_epochs),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": float(discrete_flow_shift),
        "qwen3_max_token_length": 512,
        "t5_max_token_length": 512,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size),
        "vae_disable_cache": bool(vae_disable_cache),
        "output_dir": str(output_dir),
        "output_name": project_name,
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "shuffle_caption": False,
        "caption_extension": ".txt",
        "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "training_comment": f"Anima LoRA - {datetime.now().strftime('%Y-%m-%d')}",
    }
    if logging_dir:
        training_config["log_with"] = "tensorboard"
        training_config["logging_dir"] = str(logging_dir)

    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(training_config, f)
    return str(config_path)


def create_dataset_config(project_name, image_dir, resolution=768, repeats=5, caption_dropout_rate=0.1) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_dataset_{current_date}.toml"
    dataset_config = {
        "general": {
            "resolution": int(resolution),
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "bucket_reso_steps": 64,
            "min_bucket_reso": 256,
            "max_bucket_reso": 4096,
        },
        "datasets": [
            {
                "resolution": int(resolution),
                "subsets": [
                    {
                        "num_repeats": int(repeats),
                        "image_dir": str(image_dir),
                        "caption_extension": ".txt",
                        "caption_dropout_rate": float(caption_dropout_rate),
                    }
                ],
            }
        ],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(dataset_config, f)
    return str(config_path)


# ---------------------------------------------------------------------------
# DiffSynth CLI-arg generation
# ---------------------------------------------------------------------------

def normalize_diffsynth_lora_target_modules(value: str) -> str:
    """Use DiffSynth's Anima defaults unless the user provided a custom list."""
    normalized = (value or "").strip()
    if normalized == DIFFSYNTH_LEGACY_ANIMA_TARGET_MODULES:
        return ""
    return normalized


def migrate_diffsynth_args_for_anima(args: list[str]) -> list[str]:
    """Repair saved DiffSynth arg files created by older UI versions."""
    args = list(args)
    try:
        idx = args.index("--lora_target_modules")
    except ValueError:
        pass
    else:
        value_idx = idx + 1
        if value_idx < len(args):
            args[value_idx] = normalize_diffsynth_lora_target_modules(args[value_idx])

    if "--data_file_keys" not in args:
        try:
            metadata_idx = args.index("--dataset_metadata_path")
            args[metadata_idx:metadata_idx] = ["--data_file_keys", "image"]
        except ValueError:
            args.extend(["--data_file_keys", "image"])

    try:
        metadata_idx = args.index("--dataset_metadata_path") + 1
    except ValueError:
        return args
    if metadata_idx < len(args):
        args[metadata_idx] = str(migrate_diffsynth_metadata_for_anima(Path(args[metadata_idx])))
    return args


def migrate_diffsynth_metadata_for_anima(metadata_path: Path) -> Path:
    """Convert legacy DiffSynth metadata file_name/text columns to image/prompt."""
    if not metadata_path.exists():
        return metadata_path

    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "image" in fieldnames and "prompt" in fieldnames:
        return metadata_path

    image_key = "image" if "image" in fieldnames else "file_name"
    prompt_key = "prompt" if "prompt" in fieldnames else "text"
    if image_key not in fieldnames or prompt_key not in fieldnames:
        return metadata_path

    migrated_path = metadata_path.with_name(f"{metadata_path.stem}_anima.csv")
    with open(migrated_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "prompt"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image": row.get(image_key, ""),
                "prompt": row.get(prompt_key, ""),
            })
    return migrated_path


def parse_version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value.split("+", 1)[0])
    return tuple(int(part) for part in parts[:3])


def is_version_at_most(value: str, limit: tuple[int, ...]) -> bool:
    parsed = parse_version_tuple(value)
    if not parsed:
        return False
    max_len = max(len(parsed), len(limit))
    return parsed + (0,) * (max_len - len(parsed)) <= limit + (0,) * (max_len - len(limit))


def installed_package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def create_diffsynth_training_args(
    project_name: str, output_dir: str,
    dit_model_path: Path, qwen3_model_path: Path, vae_model_path: Path,
    image_dir: str, metadata_csv: str,
    learning_rate: float, max_train_epochs: int,
    dataset_repeat: int, max_pixels: int,
    lora_rank: int, lora_target_modules: str,
    use_gradient_checkpointing: bool,
    gradient_accumulation_steps: int,
    save_steps: int,
) -> tuple[list[str], str]:
    """Build the DiffSynth CLI args list and persist them next to other configs."""
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args_path = CONFIGS_DIR / f"{project_name}_diffsynth_args_{current_date}.json"

    model_paths_json = json.dumps([str(dit_model_path), str(qwen3_model_path), str(vae_model_path)])
    lora_target_modules = normalize_diffsynth_lora_target_modules(lora_target_modules)

    args: list[str] = [
        "--dataset_base_path", str(image_dir),
        "--dataset_metadata_path", str(metadata_csv),
        "--data_file_keys", "image",
        "--max_pixels", str(int(max_pixels)),
        "--dataset_repeat", str(int(dataset_repeat)),
        "--model_paths", model_paths_json,
        "--learning_rate", str(float(learning_rate)),
        "--num_epochs", str(int(max_train_epochs)),
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--output_path", str(output_dir),
        "--lora_base_model", "dit",
        "--lora_target_modules", lora_target_modules,
        "--lora_rank", str(int(lora_rank)),
        "--gradient_accumulation_steps", str(int(gradient_accumulation_steps)),
    ]
    if use_gradient_checkpointing:
        args.append("--use_gradient_checkpointing")
    if save_steps and int(save_steps) > 0:
        args += ["--save_steps", str(int(save_steps))]

    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    return args, str(args_path)


def resolve_diffsynth_dir(cfg_value: str) -> Path:
    if cfg_value and cfg_value.strip():
        return Path(cfg_value).expanduser()
    return DIFFSYNTH_DEFAULT_DIR

def _ensure_diffsynth_installed_legacy(diffsynth_dir: Path):
    """Generator yielding log lines, ensuring `import diffsynth` works in `sys.executable`.

    Final yield is a tuple ('__done__', ok: bool, message: str).
    Clones the repo if missing, then runs `pip install -e` against the current Python.
    This protects against venv / system-Python mix-ups (e.g. Colab + .venv) where the
    user's previous setup installed DiffSynth into a different interpreter than the one
    actually launching training.
    """
    # Quick check first: maybe it's already importable in this interpreter.
    check = subprocess.run(
        [sys.executable, "-c", "import diffsynth"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        yield "✓ DiffSynth-Studio is already importable in current Python."
        yield ("__done__", True, "already installed")
        return

    yield f"DiffSynth-Studio not importable from {sys.executable} — installing now."

    # Clone if missing
    if not diffsynth_dir.exists():
        yield f"Cloning {DIFFSYNTH_GIT_URL} → {diffsynth_dir} ..."
        try:
            proc = subprocess.Popen(
                ["git", "clone", "--depth", "1", DIFFSYNTH_GIT_URL, str(diffsynth_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, encoding="utf-8", errors="ignore",
            )
        except FileNotFoundError:
            yield ("__done__", False, "git not found on PATH — install git and retry")
            return
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip("\n")
        proc.wait()
        if proc.returncode != 0:
            yield ("__done__", False, f"git clone failed (exit {proc.returncode})")
            return

    if not (diffsynth_dir / DIFFSYNTH_TRAIN_SCRIPT_REL).exists():
        yield ("__done__", False, f"Cloned dir is missing {DIFFSYNTH_TRAIN_SCRIPT_REL}")
        return

    # Editable install against the CURRENT interpreter
    yield f"Running: {sys.executable} -m pip install -e {diffsynth_dir}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-e", str(diffsynth_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1, encoding="utf-8", errors="ignore",
    )
    for line in iter(proc.stdout.readline, ""):
        yield line.rstrip("\n")
    proc.wait()
    if proc.returncode != 0:
        yield ("__done__", False, f"pip install -e failed (exit {proc.returncode})")
        return

    # Verify
    check2 = subprocess.run(
        [sys.executable, "-c", "import diffsynth; print(diffsynth.__file__)"],
        capture_output=True, text=True,
    )
    if check2.returncode != 0:
        yield ("__done__", False, f"After install, `import diffsynth` still fails:\n{check2.stderr}")
        return
    yield f"✓ DiffSynth installed at: {check2.stdout.strip()}"
    yield ("__done__", True, "installed")


def ensure_diffsynth_installed(diffsynth_dir: Path):
    """Ensure a local DiffSynth-Studio checkout and an importable package exist."""
    if diffsynth_dir.exists() and not diffsynth_dir.is_dir():
        yield ("__done__", False, f"{diffsynth_dir} exists but is not a directory")
        return

    ds_script = diffsynth_dir / DIFFSYNTH_TRAIN_SCRIPT_REL

    def stream_process(cmd: list[str], cwd: str | None = None):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=cwd,
            encoding="utf-8",
            errors="ignore",
        )
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip("\n")
        proc.wait()
        return proc.returncode

    def ensure_torchao_compatible():
        version = installed_package_version("torchao")
        if version is None:
            return True, "torchao not installed"
        if not is_version_at_most(version, TORCHAO_MIN_EXCLUSIVE_VERSION):
            yield f"torchao {version} is compatible with PEFT."
            return True, "torchao compatible"

        yield (
            f"torchao {version} is incompatible with current PEFT; "
            f"upgrading to {TORCHAO_PIP_SPEC} ..."
        )
        returncode = yield from stream_process(
            [sys.executable, "-m", "pip", "install", "--upgrade", TORCHAO_PIP_SPEC]
        )
        if returncode == 0:
            new_version = installed_package_version("torchao") or "unknown"
            yield f"torchao upgraded to {new_version}."
            return True, "torchao upgraded"

        yield "torchao upgrade failed; uninstalling incompatible torchao so PEFT can use standard LoRA dispatch."
        returncode = yield from stream_process(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]
        )
        if returncode != 0:
            return False, f"torchao upgrade/uninstall failed (exit {returncode})"
        yield "Incompatible torchao removed."
        return True, "torchao removed"

    def clone_repo():
        yield f"Cloning {DIFFSYNTH_GIT_URL} -> {diffsynth_dir} ..."
        try:
            diffsynth_dir.parent.mkdir(parents=True, exist_ok=True)
            returncode = yield from stream_process(
                ["git", "clone", "--depth", "1", DIFFSYNTH_GIT_URL, str(diffsynth_dir)]
            )
        except FileNotFoundError:
            yield ("__done__", False, "git not found on PATH - install git and retry")
            return
        if returncode != 0:
            yield ("__done__", False, f"git clone failed (exit {returncode})")

    if not ds_script.exists():
        if not diffsynth_dir.exists() or not any(diffsynth_dir.iterdir()):
            for item in clone_repo():
                if isinstance(item, tuple):
                    yield item
                    return
                yield item
        elif (diffsynth_dir / ".git").exists():
            yield f"DiffSynth-Studio exists but Anima train script is missing; updating {diffsynth_dir} ..."
            try:
                returncode = yield from stream_process(["git", "pull", "--ff-only"], cwd=str(diffsynth_dir))
            except FileNotFoundError:
                yield ("__done__", False, "git not found on PATH - install git and retry")
                return
            if returncode != 0:
                yield ("__done__", False, f"git pull failed (exit {returncode})")
                return
        else:
            yield (
                "__done__",
                False,
                f"{diffsynth_dir} exists but is missing {DIFFSYNTH_TRAIN_SCRIPT_REL}. "
                "Choose an empty directory or a DiffSynth-Studio git clone.",
            )
            return

    ds_script = diffsynth_dir / DIFFSYNTH_TRAIN_SCRIPT_REL
    if not ds_script.exists():
        yield ("__done__", False, f"DiffSynth-Studio is missing {DIFFSYNTH_TRAIN_SCRIPT_REL}")
        return

    check = subprocess.run(
        [sys.executable, "-c", "import diffsynth; print(diffsynth.__file__)"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        imported_path = Path(check.stdout.strip()).resolve()
        try:
            imported_path.relative_to(diffsynth_dir.resolve())
            yield f"DiffSynth-Studio is already installed from: {imported_path}"
            torchao_ok, torchao_msg = yield from ensure_torchao_compatible()
            if not torchao_ok:
                yield ("__done__", False, torchao_msg)
                return
            yield ("__done__", True, "already installed")
            return
        except ValueError:
            yield f"Found diffsynth at {imported_path}; reinstalling local DiffSynth-Studio checkout."

    if check.returncode != 0:
        yield f"DiffSynth-Studio not importable from {sys.executable} - installing now."
    yield f"Running: {sys.executable} -m pip install -e {diffsynth_dir}"
    try:
        returncode = yield from stream_process([sys.executable, "-m", "pip", "install", "-e", str(diffsynth_dir)])
    except FileNotFoundError:
        yield ("__done__", False, "python or pip not found")
        return
    if returncode != 0:
        yield ("__done__", False, f"pip install -e failed (exit {returncode})")
        return

    check2 = subprocess.run(
        [sys.executable, "-c", "import diffsynth; print(diffsynth.__file__)"],
        capture_output=True,
        text=True,
    )
    if check2.returncode != 0:
        yield ("__done__", False, f"After install, import diffsynth still fails:\n{check2.stderr}")
        return
    yield f"DiffSynth installed at: {check2.stdout.strip()}"
    torchao_ok, torchao_msg = yield from ensure_torchao_compatible()
    if not torchao_ok:
        yield ("__done__", False, torchao_msg)
        return
    yield ("__done__", True, "installed")


# ---------------------------------------------------------------------------
# Configure Training handler
# ---------------------------------------------------------------------------

def configure_training(
    backend, diffsynth_dir,
    project_name, base_model, image_directory, output_directory,
    network_dim, network_alpha, learning_rate, max_train_epochs,
    resolution, repeats, caption_dropout, gpu_index_choice,
    # advanced (kohya)
    optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
    train_batch_size, gradient_accumulation_steps, max_grad_norm,
    save_every_n_epochs, save_last_n_epochs, mixed_precision,
    gradient_checkpointing, seed, noise_offset, multires_noise_discount,
    timestep_sampling, discrete_flow_shift,
    cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
    num_cpu_threads_per_process,
    # DiffSynth-specific
    lora_target_modules, dataset_repeat, max_pixels, save_steps_ds,
    # TensorBoard
    use_tensorboard, tb_logdir_input, tb_port,
) -> tuple[str, str, str, str, str]:
    """
    Returns (status_message, last_train_config_path, last_dataset_config_path,
             last_diffsynth_args_path, last_tb_logdir).
    """
    lines = []
    backend = (backend or "kohya").lower()

    # --- Validate inputs ---
    if not project_name.strip():
        return t("err_project_empty"), "", "", "", ""
    if not image_directory.strip():
        return t("err_image_dir_empty"), "", "", "", ""
    if not output_directory.strip():
        return t("err_output_dir_empty"), "", "", "", ""

    lines.append(t("info_backend", backend=backend))
    lines.append(t("info_project", name=project_name))
    lines.append(t("info_image_dir", dir=image_directory))
    lines.append(t("info_output_dir", dir=output_directory))
    lines.append("")

    # --- Validate dataset ---
    try:
        n_images, missing, warnings = validate_dataset(image_directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        return f"❌ {e}", "", "", "", ""

    lines.append(t("info_images_found", n=n_images))
    if missing:
        lines.append(t("info_missing_captions", n=len(missing)))
        for m in missing[:20]:
            lines.append(f"    • {m}")
        if len(missing) > 20:
            lines.append(t("info_more", n=len(missing) - 20))
    else:
        lines.append(t("info_all_have_captions"))

    for w in warnings:
        lines.append(f"⚠ {w}")

    if n_images == 0:
        lines.append("")
        lines.append(t("info_no_images"))
        return "\n".join(lines), "", "", "", ""

    # --- Step estimate ---
    batch = max(int(train_batch_size), 1)
    grad = max(int(gradient_accumulation_steps), 1)
    effective_repeats = int(repeats) if backend == "kohya" else int(dataset_repeat)
    spe = math.ceil((n_images * effective_repeats) / (batch * grad))
    total = spe * int(max_train_epochs)
    lines.append("")
    lines.append(t("info_step_header"))
    lines.append(t("info_step_per_epoch", n=spe, imgs=n_images, repeats=effective_repeats))
    lines.append(t("info_step_total", n=total, spe=spe, ep=int(max_train_epochs)))
    lines.append(t("info_step_footer"))

    # --- Validate models ---
    lines.append("")
    lines.append(t("info_checking_models"))
    dit_model = get_dit_model_path(base_model)
    missing_models = []
    for label, path in [("DiT", dit_model), ("Qwen3", QWEN3_MODEL), ("VAE", VAE_MODEL)]:
        if Path(path).exists():
            lines.append(f"  ✓ {label}: {path}")
        else:
            if label == "DiT":
                lines.append(f"  ℹ {label}: {path}")
                lines.append(t("info_will_download"))
            else:
                lines.append(f"  ✗ {label} missing: {path}")
                missing_models.append(label)
    if missing_models:
        lines.append("")
        lines.append(t("info_missing_models", list=", ".join(missing_models)))
        lines.append(t("info_run_setup"))
        return "\n".join(lines), "", "", "", ""

    # --- Resolve TensorBoard logdir ---
    tb_logdir = ""
    if use_tensorboard:
        candidate = (tb_logdir_input or "").strip()
        if not candidate:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            candidate = str(TB_LOGS_ROOT / f"{project_name}_{ts}")
        Path(candidate).mkdir(parents=True, exist_ok=True)
        tb_logdir = candidate
        lines.append("")
        lines.append(t("info_tb_enabled", dir=tb_logdir))

    # --- Backend-specific config generation ---
    lines.append("")
    train_cfg = ""
    dataset_cfg = ""
    diffsynth_args_path = ""

    if backend == "kohya":
        lines.append(t("info_generating_toml"))
        try:
            train_cfg = create_kohya_training_config(
                project_name=project_name, output_dir=output_directory,
                dit_model_path=dit_model, qwen3_model_path=QWEN3_MODEL, vae_model_path=VAE_MODEL,
                network_dim=network_dim, network_alpha=network_alpha,
                learning_rate=learning_rate, max_train_epochs=max_train_epochs,
                optimizer_type=optimizer_type, lr_scheduler=lr_scheduler,
                lr_scheduler_num_cycles=lr_scheduler_num_cycles, lr_warmup_steps=lr_warmup_steps,
                train_batch_size=train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                save_every_n_epochs=save_every_n_epochs, save_last_n_epochs=save_last_n_epochs,
                mixed_precision=mixed_precision, gradient_checkpointing=gradient_checkpointing,
                seed=seed, noise_offset=noise_offset, multires_noise_discount=multires_noise_discount,
                timestep_sampling=timestep_sampling, discrete_flow_shift=discrete_flow_shift,
                cache_latents=cache_latents, cache_text_encoder_outputs=cache_text_encoder_outputs,
                vae_chunk_size=vae_chunk_size, vae_disable_cache=vae_disable_cache,
                logging_dir=tb_logdir,
            )
            dataset_cfg = create_dataset_config(
                project_name=project_name, image_dir=image_directory,
                resolution=resolution, repeats=repeats, caption_dropout_rate=caption_dropout,
            )
        except Exception as e:
            lines.append(t("err_generate_failed", err=e))
            return "\n".join(lines), "", "", "", ""

        lines.append(t("info_train_cfg_written", path=train_cfg))
        lines.append(t("info_dataset_cfg_written", path=dataset_cfg))

    elif backend == "diffsynth":
        # Soft check — if DiffSynth-Studio isn't cloned yet, just inform the user.
        # Actual install happens at training time via ensure_diffsynth_installed().
        ds_dir = resolve_diffsynth_dir(diffsynth_dir)
        ds_script = ds_dir / DIFFSYNTH_TRAIN_SCRIPT_REL
        if not ds_script.exists():
            lines.append("")
            lines.append(t("info_diffsynth_will_install", path=str(ds_dir)))

        lines.append(t("info_generating_metadata"))
        try:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            metadata_path = CONFIGS_DIR / f"{project_name}_metadata_{ts}.csv"
            metadata_path, n_rows = generate_diffsynth_metadata(image_directory, metadata_path)
            lines.append(t("info_metadata_written", path=str(metadata_path), n=n_rows))

            _, diffsynth_args_path = create_diffsynth_training_args(
                project_name=project_name,
                output_dir=output_directory,
                dit_model_path=dit_model,
                qwen3_model_path=QWEN3_MODEL,
                vae_model_path=VAE_MODEL,
                image_dir=image_directory,
                metadata_csv=str(metadata_path),
                learning_rate=learning_rate,
                max_train_epochs=max_train_epochs,
                dataset_repeat=dataset_repeat,
                max_pixels=max_pixels,
                lora_rank=network_dim,
                lora_target_modules=lora_target_modules,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_accumulation_steps=gradient_accumulation_steps,
                save_steps=save_steps_ds,
            )
            lines.append(t("info_args_written", path=diffsynth_args_path))
        except Exception as e:
            lines.append(t("err_generate_failed", err=e))
            return "\n".join(lines), "", "", "", ""

    else:
        lines.append(f"❌ Unknown backend: {backend}")
        return "\n".join(lines), "", "", "", ""

    # --- Save all settings to config.json ---
    cfg = {
        "backend": backend,
        "diffsynth_dir": diffsynth_dir or "",
        "use_tensorboard": bool(use_tensorboard),
        "tb_port": int(tb_port),
        "tb_logdir": tb_logdir,
        "lora_target_modules": normalize_diffsynth_lora_target_modules(lora_target_modules),
        "dataset_repeat": int(dataset_repeat),
        "max_pixels": int(max_pixels),
        "save_steps_ds": int(save_steps_ds),
        "project_name": project_name,
        "base_model": base_model,
        "image_directory": image_directory,
        "output_directory": output_directory,
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "learning_rate": float(learning_rate),
        "max_train_epochs": int(max_train_epochs),
        "resolution": int(resolution),
        "repeats": int(repeats),
        "caption_dropout": float(caption_dropout),
        "gpu_index": gpu_index_from_choice(gpu_index_choice),
        "optimizer_type": optimizer_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "seed": int(seed),
        "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": float(discrete_flow_shift),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size),
        "vae_disable_cache": bool(vae_disable_cache),
        "num_cpu_threads_per_process": int(num_cpu_threads_per_process),
        "last_train_config": train_cfg,
        "last_dataset_config": dataset_cfg,
        "last_diffsynth_args": diffsynth_args_path,
        "last_tb_logdir": tb_logdir,
    }
    save_config(cfg)

    lines.append("")
    lines.append(t("info_ready"))
    return "\n".join(lines), train_cfg, dataset_cfg, diffsynth_args_path, tb_logdir


# ---------------------------------------------------------------------------
# DiffSynth stdout → TensorBoard loss writer
# ---------------------------------------------------------------------------

_LOSS_RE = re.compile(r"loss[=:\s]+([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")


class DiffSynthLossWriter:
    """Parse tqdm/stdout loss values and stream them to a TF events file."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.step = 0
        self.writer = None
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except Exception as e:
            print(f"[DiffSynthLossWriter] tensorboard unavailable: {e}", file=sys.stderr)

    def feed(self, line: str):
        if self.writer is None:
            return
        m = _LOSS_RE.search(line)
        if m:
            try:
                value = float(m.group(1))
                if 0.0 < value < 1e6:  # filter junk matches
                    self.writer.add_scalar("loss", value, self.step)
                    self.step += 1
            except ValueError:
                pass

    def close(self):
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Training runner (generator — streams logs live to Gradio)
# ---------------------------------------------------------------------------

def start_training(
    backend: str,
    diffsynth_dir: str,
    custom_config_path: str,
    gpu_index_choice: str,
    num_cpu_threads_per_process: int,
    base_model: str,
    use_tensorboard: bool,
):
    """Generator: yields growing log text as training runs."""
    log_lines: list[str] = []
    backend = (backend or "kohya").lower()

    def emit(line: str):
        log_lines.append(line)
        return "\n".join(log_lines)

    # --- Auto-download DiT model if needed ---
    dit_model = get_dit_model_path(base_model)
    if not dit_model.exists():
        url = BASE_MODEL_URLS.get(base_model)
        if not url:
            yield emit(t("err_unknown_base_model", name=base_model))
            return
        yield emit(t("info_downloading_model", name=base_model))
        yield emit(t("info_download_destination", path=str(dit_model)))
        yield emit("")
        os.makedirs(dit_model.parent, exist_ok=True)
        try:
            dl_proc = subprocess.Popen(
                ["wget", "-c", "--show-progress", "-O", str(dit_model), url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            for line in iter(dl_proc.stdout.readline, ""):
                yield emit(line.rstrip("\n"))
            dl_proc.wait()
            if dl_proc.returncode != 0:
                yield emit(t("err_download_failed", code=dl_proc.returncode))
                return
            yield emit(t("info_download_done"))
            yield emit("")
        except FileNotFoundError:
            yield emit(t("err_wget_missing"))
            return

    saved_cfg = load_config()
    threads = max(int(num_cpu_threads_per_process), 1)
    gpu_idx = gpu_index_from_choice(gpu_index_choice)
    tb_logdir = saved_cfg.get("last_tb_logdir", "")
    accelerate_launch = resolve_accelerate_launch_cmd()

    # --- Backend-specific command assembly ---
    if backend == "kohya":
        train_cfg = custom_config_path.strip() if custom_config_path.strip() else saved_cfg.get("last_train_config", "")
        dataset_cfg = saved_cfg.get("last_dataset_config", "")

        if not train_cfg:
            yield emit(t("err_no_train_cfg"))
            return
        if not Path(train_cfg).exists():
            yield emit(t("err_train_cfg_not_found", path=train_cfg))
            return
        if not dataset_cfg:
            yield emit(t("err_no_dataset_cfg"))
            return
        if not Path(dataset_cfg).exists():
            yield emit(t("err_dataset_cfg_not_found", path=dataset_cfg))
            return
        if not TRAIN_SCRIPT.exists():
            yield emit(t("err_train_script_missing", path=str(TRAIN_SCRIPT)))
            return

        cmd = [
            *accelerate_launch,
            "--config_file", str(ACCELERATE_CONFIG),
            "--num_cpu_threads_per_process", str(threads),
            "--gpu_ids", gpu_idx,
            str(TRAIN_SCRIPT),
            "--config_file", train_cfg,
            "--dataset_config", dataset_cfg,
        ]
        loss_writer = None
        cwd = str(ROOT)

    elif backend == "diffsynth":
        diffsynth_args_path = saved_cfg.get("last_diffsynth_args", "")
        if not diffsynth_args_path or not Path(diffsynth_args_path).exists():
            yield emit(t("err_no_train_cfg"))
            return

        try:
            with open(diffsynth_args_path, "r", encoding="utf-8") as f:
                ds_args = json.load(f)
            ds_args = migrate_diffsynth_args_for_anima(ds_args)
        except Exception as e:
            yield emit(t("err_generate_failed", err=e))
            return

        ds_dir = resolve_diffsynth_dir(diffsynth_dir or saved_cfg.get("diffsynth_dir", ""))

        # Make sure DiffSynth is installed in *this* interpreter — auto-clones
        # and pip-installs if missing. Streams progress to the log box.
        yield emit(t("info_diffsynth_check"))
        install_ok = False
        install_msg = ""
        for item in ensure_diffsynth_installed(ds_dir):
            if isinstance(item, tuple) and item and item[0] == "__done__":
                _, install_ok, install_msg = item
            else:
                yield emit(item)
        if not install_ok:
            yield emit(t("err_diffsynth_install_failed", err=install_msg))
            return

        ds_script = ds_dir / DIFFSYNTH_TRAIN_SCRIPT_REL
        if not ds_script.exists():
            yield emit(t("err_diffsynth_missing", path=str(ds_dir)))
            return

        cmd = [
            *accelerate_launch,
            "--config_file", str(ACCELERATE_CONFIG),
            "--num_cpu_threads_per_process", str(threads),
            "--gpu_ids", gpu_idx,
            str(ds_script),
            *ds_args,
        ]
        loss_writer = DiffSynthLossWriter(tb_logdir) if use_tensorboard and tb_logdir else None
        cwd = str(ds_dir)

    else:
        yield emit(f"❌ Unknown backend: {backend}")
        return

    yield emit(t("info_using_gpu", idx=gpu_idx))
    yield emit(t("info_command", cmd=" ".join(shlex.quote(c) for c in cmd)))
    yield emit("")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    project_name = saved_cfg.get("project_name", "run")
    log_file_path = LOGS_DIR / f"{project_name}_{backend}_{timestamp}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            cwd=cwd,
            encoding="utf-8",
            errors="ignore",
        )
    except FileNotFoundError:
        yield emit(t("err_accelerate_missing"))
        if loss_writer:
            loss_writer.close()
        return

    with open(log_file_path, "w", encoding="utf-8", errors="ignore") as log_f:
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Started: {datetime.now().isoformat()}\n\n")
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip("\n")
            log_f.write(line + "\n")
            log_f.flush()
            if loss_writer:
                loss_writer.feed(line)
            yield emit(line)

    exit_code = process.wait()
    if loss_writer:
        loss_writer.close()

    if exit_code == 0:
        yield emit(t("info_train_done", output=saved_cfg.get("output_directory", "output dir"), log=str(log_file_path)))
    else:
        yield emit(t("info_train_failed", code=exit_code, log=str(log_file_path)))
        try:
            result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
            tail = "\n".join(result.stdout.splitlines()[-40:])
            if any(s in tail for s in ("Out of memory", "Killed process", "oom_reaper", "OOM")):
                yield emit(t("info_oom_hint"))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TensorBoard server management
# ---------------------------------------------------------------------------

_tb_proc: subprocess.Popen | None = None
_tb_lock = threading.Lock()
_ngrok_tunnels: dict[int, object] = {}
_ngrok_lock = threading.Lock()


def _port_alive(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def start_ngrok_tunnel(port: int, token: str) -> tuple[str | None, str | None]:
    """Open (or reuse) an ngrok HTTPS tunnel to a local port. Returns (public_url, error)."""
    if not PYNGROK_AVAILABLE:
        return None, t("ngrok_status_no_pyngrok")
    token = (token or "").strip() or os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        return None, t("ngrok_status_no_token")

    with _ngrok_lock:
        try:
            _ngrok.set_auth_token(token)
            old = _ngrok_tunnels.pop(int(port), None)
            if old is not None:
                try:
                    _ngrok.disconnect(old.public_url)
                except Exception:
                    pass
            tunnel = _ngrok.connect(int(port), "http", bind_tls=True)
            _ngrok_tunnels[int(port)] = tunnel
            url = tunnel.public_url
            if url.startswith("http://"):
                url = "https://" + url[len("http://"):]
            return url, None
        except Exception as e:
            return None, t("ngrok_status_tunnel_failed", err=e)


def stop_ngrok_tunnels():
    if not PYNGROK_AVAILABLE:
        return
    with _ngrok_lock:
        for port, tunnel in list(_ngrok_tunnels.items()):
            try:
                _ngrok.disconnect(tunnel.public_url)
            except Exception:
                pass
            _ngrok_tunnels.pop(port, None)
        try:
            _ngrok.kill()
        except Exception:
            pass


atexit.register(stop_ngrok_tunnels)


def start_tensorboard(
    log_dir: str, port: int,
    use_ngrok: bool = False, ngrok_token: str = "",
) -> tuple[str, str]:
    """Returns (status_message, iframe_html). When use_ngrok is True, also opens an ngrok tunnel."""
    global _tb_proc
    port = int(port) if port else 6006

    if not log_dir or not log_dir.strip():
        return t("tb_status_no_logdir"), _empty_iframe()
    if not Path(log_dir).exists():
        return t("tb_status_no_logdir"), _empty_iframe()

    # When tunneling, TB must bind to all interfaces so ngrok can reach it.
    bind_host = "0.0.0.0" if use_ngrok else "127.0.0.1"

    with _tb_lock:
        if _tb_proc is None or _tb_proc.poll() is not None:
            try:
                _tb_proc = subprocess.Popen(
                    [sys.executable, "-m", "tensorboard.main",
                     "--logdir", log_dir,
                     "--port", str(port),
                     "--host", bind_host,
                     "--reload_interval", "5"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError as e:
                return t("tb_status_failed", err=e), _empty_iframe()

            for _ in range(60):
                if _port_alive(port):
                    break
                time.sleep(0.5)
            else:
                return t("tb_status_failed", err="timeout waiting for port"), _empty_iframe()
            already_running = False
        else:
            already_running = True

    local_url = f"http://127.0.0.1:{port}"
    status_lines: list[str] = []
    if already_running:
        status_lines.append(t("tb_status_already", url=local_url))
    else:
        status_lines.append(t("tb_status_running", url=local_url))

    if use_ngrok:
        public_url, err = start_ngrok_tunnel(port, ngrok_token)
        if public_url:
            status_lines.append(t("ngrok_status_tunnel_open", url=public_url))
            return "\n\n".join(status_lines), _iframe_for(public_url, local_url)
        else:
            status_lines.append(err or t("tb_status_failed", err="ngrok"))
            # Fall back to local URL — useful when running locally even if user accidentally toggled ngrok
            return "\n\n".join(status_lines), _iframe_for(local_url)

    return "\n\n".join(status_lines), _iframe_for(local_url)


def stop_tensorboard() -> tuple[str, str]:
    global _tb_proc
    with _tb_lock:
        if _tb_proc is not None and _tb_proc.poll() is None:
            _tb_proc.terminate()
            try:
                _tb_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _tb_proc.kill()
        _tb_proc = None
    stop_ngrok_tunnels()
    return t("tb_status_stopped"), _empty_iframe()


def _iframe_for(url: str, alt_url: str | None = None) -> str:
    alt_block = ""
    if alt_url and alt_url != url:
        alt_block = (
            f' · <a href="{alt_url}" target="_blank" rel="noopener noreferrer">'
            f'{t("btn_open_local")}</a>'
        )
    return (
        f'<div style="border:1px solid #ccc;border-radius:6px;overflow:hidden;">'
        f'<iframe src="{url}" width="100%" height="800" style="border:0;"></iframe>'
        f'<div style="padding:6px;font-size:13px;">'
        f'🔗 <a href="{url}" target="_blank" rel="noopener noreferrer">{t("btn_open_tb")}</a>'
        f'{alt_block}'
        f'</div></div>'
    )


def _empty_iframe() -> str:
    return f'<div style="padding:30px;text-align:center;color:#888;">{t("tb_view_placeholder")}</div>'


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    cfg = load_config()
    current_lang = get_lang()

    saved_gpu_idx = str(cfg.get("gpu_index", "0"))
    default_gpu = next(
        (c for c in GPU_CHOICES if c.startswith(saved_gpu_idx + ":")),
        GPU_CHOICES[0] if GPU_CHOICES else "0",
    )

    is_diffsynth = cfg.get("backend", "kohya") == "diffsynth"

    with gr.Blocks(title=t("app_title")) as demo:
        gr.Markdown(t("header_markdown"))

        # ── Language + Backend top bar ───────────────────────────────────
        with gr.Row():
            language_dd = gr.Dropdown(
                label=t("language"),
                choices=SUPPORTED_LANGS,
                value=current_lang,
                info=t("language_info"),
                scale=1,
            )
            save_lang_btn = gr.Button(t("btn_save_language"), scale=0)
            backend_radio = gr.Radio(
                label=t("backend"),
                choices=["kohya", "diffsynth"],
                value=cfg.get("backend", "kohya"),
                info=t("backend_info"),
                scale=2,
            )

        language_status = gr.Markdown("")

        # ── Shared state for last-generated config paths ────────────────
        last_train_cfg = gr.State(cfg.get("last_train_config", ""))
        last_dataset_cfg = gr.State(cfg.get("last_dataset_config", ""))
        last_diffsynth_args = gr.State(cfg.get("last_diffsynth_args", ""))
        last_tb_logdir_state = gr.State(cfg.get("last_tb_logdir", ""))

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Training
            # ================================================================
            with gr.Tab(t("tab_training")):

                with gr.Group():
                    gr.Markdown(f"### {t('section_project_paths')}")
                    with gr.Row():
                        project_name = gr.Textbox(label=t("project_name"), value=cfg["project_name"], placeholder="my_lora")
                        gpu_dropdown = gr.Dropdown(label=t("gpu"), choices=GPU_CHOICES, value=default_gpu)
                    with gr.Row():
                        base_model_dropdown = gr.Dropdown(
                            label=t("base_model"),
                            choices=["anima-base-v1.0", "anima-preview3-base", "anima-preview"],
                            value=cfg.get("base_model", "anima-base-v1.0"),
                            info=t("base_model_info"),
                        )
                    image_directory = gr.Textbox(
                        label=t("image_directory"),
                        value=cfg["image_directory"],
                        placeholder="/path/to/my_dataset",
                    )
                    output_directory = gr.Textbox(
                        label=t("output_directory"),
                        value=cfg["output_directory"],
                        placeholder="/path/to/output",
                    )

                with gr.Group():
                    gr.Markdown(f"### {t('section_network')}")
                    with gr.Row():
                        network_dim = gr.Number(label=t("network_dim"), value=cfg["network_dim"], precision=0, minimum=1)
                        network_alpha = gr.Number(label=t("network_alpha"), value=cfg["network_alpha"], precision=0, minimum=1)
                        learning_rate = gr.Number(label=t("learning_rate"), value=cfg["learning_rate"])
                        max_train_epochs = gr.Number(label=t("max_epochs"), value=cfg["max_train_epochs"], precision=0, minimum=1)

                with gr.Group():
                    gr.Markdown(f"### {t('section_dataset')}")
                    with gr.Row():
                        resolution = gr.Number(label=t("resolution"), value=cfg["resolution"], precision=0, minimum=64)
                        repeats = gr.Number(label=t("repeats"), value=cfg["repeats"], precision=0, minimum=1)
                        caption_dropout = gr.Slider(label=t("caption_dropout"), minimum=0.0, maximum=1.0, step=0.05, value=cfg["caption_dropout"])

                gr.Markdown("---")
                gr.Markdown(f"### {t('section_config_training')}")

                with gr.Row():
                    configure_btn = gr.Button(t("btn_configure"), variant="secondary", size="lg")
                    train_btn = gr.Button(t("btn_start"), variant="primary", size="lg")

                custom_config_input = gr.Textbox(
                    label=t("override_config_label"),
                    value="",
                    placeholder="/path/to/custom_training_config.toml",
                )

                status_box = gr.Textbox(label=t("status_label"), lines=14, interactive=False, show_copy_button=True)
                log_box = gr.Textbox(label=t("log_label"), lines=25, interactive=False, show_copy_button=True, autoscroll=True)

            # ================================================================
            # TAB 2 — Advanced Settings
            # ================================================================
            with gr.Tab(t("tab_advanced")):
                gr.Markdown(
                    "_These settings are applied when you click "
                    f"**{t('btn_configure')}**._"
                )

                # DiffSynth-specific group (visible only when backend == diffsynth)
                with gr.Group(visible=is_diffsynth) as diffsynth_group:
                    gr.Markdown(f"### {t('section_diffsynth')}")
                    gr.Markdown(t("diffsynth_lr_note"))
                    with gr.Row():
                        lora_target_modules_in = gr.Textbox(
                            label=t("lora_target_modules"),
                            value=cfg["lora_target_modules"],
                            info=t("lora_target_modules_info"),
                        )
                    with gr.Row():
                        dataset_repeat_in = gr.Number(label=t("dataset_repeat"), value=cfg["dataset_repeat"], precision=0, minimum=1)
                        max_pixels_in = gr.Number(label=t("max_pixels"), value=cfg["max_pixels"], precision=0, minimum=65536)
                        save_steps_ds_in = gr.Number(
                            label=t("save_steps_ds"), value=cfg["save_steps_ds"],
                            precision=0, minimum=0, info=t("save_steps_ds_info"),
                        )
                    diffsynth_dir_in = gr.Textbox(
                        label=t("diffsynth_dir"),
                        value=cfg["diffsynth_dir"],
                        placeholder=str(DIFFSYNTH_DEFAULT_DIR),
                        info=t("diffsynth_dir_info"),
                    )

                # Kohya-specific groups (visible only when backend == kohya)
                with gr.Group(visible=not is_diffsynth) as kohya_optimizer_group:
                    gr.Markdown(f"### {t('section_optimizer')}")
                    with gr.Row():
                        optimizer_type = gr.Dropdown(
                            label=t("optimizer"),
                            choices=["AdamW8bit", "AdamW", "Lion", "SGD", "Prodigy"],
                            value=cfg["optimizer_type"],
                        )
                        lr_scheduler = gr.Dropdown(
                            label=t("lr_scheduler"),
                            choices=["cosine_with_restarts", "cosine", "linear", "constant", "constant_with_warmup", "polynomial"],
                            value=cfg["lr_scheduler"],
                        )
                    with gr.Row():
                        lr_scheduler_num_cycles = gr.Number(label=t("lr_scheduler_cycles"), value=cfg["lr_scheduler_num_cycles"], precision=0, minimum=1)
                        lr_warmup_steps = gr.Number(label=t("lr_warmup_steps"), value=cfg["lr_warmup_steps"], precision=0, minimum=0)

                with gr.Group():
                    gr.Markdown(f"### {t('section_batch')}")
                    with gr.Row():
                        train_batch_size = gr.Number(label=t("train_batch_size"), value=cfg["train_batch_size"], precision=0, minimum=1)
                        gradient_accumulation_steps = gr.Number(label=t("grad_accum_steps"), value=cfg["gradient_accumulation_steps"], precision=0, minimum=1)
                        max_grad_norm = gr.Number(label=t("max_grad_norm"), value=cfg["max_grad_norm"])

                with gr.Group(visible=not is_diffsynth) as kohya_saving_group:
                    gr.Markdown(f"### {t('section_saving')}")
                    with gr.Row():
                        save_every_n_epochs = gr.Number(label=t("save_every_n_epochs"), value=cfg["save_every_n_epochs"], precision=0, minimum=1)
                        save_last_n_epochs = gr.Number(label=t("save_last_n"), value=cfg["save_last_n_epochs"], precision=0, minimum=1)

                with gr.Group():
                    gr.Markdown(f"### {t('section_precision')}")
                    with gr.Row():
                        mixed_precision = gr.Dropdown(label=t("mixed_precision"), choices=["bf16", "fp16", "no"], value=cfg["mixed_precision"])
                        vae_chunk_size = gr.Number(label=t("vae_chunk_size"), value=cfg["vae_chunk_size"], precision=0, minimum=1, visible=not is_diffsynth)
                    with gr.Row():
                        gradient_checkpointing = gr.Checkbox(label=t("gradient_checkpointing"), value=cfg["gradient_checkpointing"])
                        cache_latents = gr.Checkbox(label=t("cache_latents"), value=cfg["cache_latents"], visible=not is_diffsynth)
                        cache_text_encoder_outputs = gr.Checkbox(label=t("cache_text_encoder"), value=cfg["cache_text_encoder_outputs"], visible=not is_diffsynth)
                        vae_disable_cache = gr.Checkbox(label=t("vae_disable_cache"), value=cfg["vae_disable_cache"], visible=not is_diffsynth)

                with gr.Group(visible=not is_diffsynth) as kohya_noise_group:
                    gr.Markdown(f"### {t('section_noise')}")
                    with gr.Row():
                        noise_offset = gr.Number(label=t("noise_offset"), value=cfg["noise_offset"])
                        multires_noise_discount = gr.Number(label=t("multires_noise_discount"), value=cfg["multires_noise_discount"])
                        timestep_sampling = gr.Dropdown(label=t("timestep_sampling"), choices=["sigmoid", "uniform", "logit_normal"], value=cfg["timestep_sampling"])
                        discrete_flow_shift = gr.Number(label=t("discrete_flow_shift"), value=cfg["discrete_flow_shift"])

                with gr.Group():
                    gr.Markdown(f"### {t('section_misc')}")
                    with gr.Row():
                        seed = gr.Number(label=t("seed"), value=cfg["seed"], precision=0)
                        num_cpu_threads = gr.Number(label=t("cpu_threads"), value=cfg["num_cpu_threads_per_process"], precision=0, minimum=1)

            # ================================================================
            # TAB 3 — TensorBoard
            # ================================================================
            with gr.Tab(t("tab_tensorboard")):
                gr.Markdown(f"### {t('section_tb_settings')}")
                with gr.Row():
                    use_tb_chk = gr.Checkbox(label=t("tb_use"), value=cfg["use_tensorboard"], info=t("tb_use_info"))
                    tb_port_in = gr.Number(label=t("tb_port"), value=cfg["tb_port"], precision=0, minimum=1024)
                tb_logdir_in = gr.Textbox(
                    label=t("tb_logdir"),
                    value=cfg.get("last_tb_logdir", "") or cfg.get("tb_logdir", ""),
                    info=t("tb_logdir_info"),
                    placeholder=str(TB_LOGS_ROOT / "<project>_<timestamp>"),
                )

                with gr.Group():
                    gr.Markdown(f"### {t('section_sharing')}")
                    if IS_COLAB:
                        gr.Markdown(t("info_colab_detected"))
                    if not PYNGROK_AVAILABLE:
                        gr.Markdown(t("ngrok_status_no_pyngrok"))
                    ngrok_enable_chk = gr.Checkbox(
                        label=t("ngrok_enable"),
                        value=bool(cfg.get("ngrok_enable", False)) or IS_COLAB,
                        info=t("ngrok_enable_info"),
                    )
                    ngrok_token_in = gr.Textbox(
                        label=t("ngrok_token"),
                        value=cfg.get("ngrok_token", ""),
                        type="password",
                        info=t("ngrok_token_info"),
                        placeholder="2x...your token...",
                    )

                with gr.Row():
                    start_tb_btn = gr.Button(t("btn_start_tb"), variant="primary")
                    stop_tb_btn = gr.Button(t("btn_stop_tb"), variant="stop")
                tb_status_md = gr.Markdown(t("tb_status_stopped"))
                tb_iframe = gr.HTML(_empty_iframe())

        # ── Backend visibility toggling ──────────────────────────────────
        def _toggle_backend(backend_value: str):
            ds = backend_value == "diffsynth"
            return (
                gr.update(visible=ds),         # diffsynth_group
                gr.update(visible=not ds),     # kohya_optimizer_group
                gr.update(visible=not ds),     # kohya_saving_group
                gr.update(visible=not ds),     # kohya_noise_group
                gr.update(visible=not ds),     # vae_chunk_size
                gr.update(visible=not ds),     # cache_latents
                gr.update(visible=not ds),     # cache_text_encoder
                gr.update(visible=not ds),     # vae_disable_cache
            )

        backend_radio.change(
            fn=_toggle_backend,
            inputs=[backend_radio],
            outputs=[
                diffsynth_group, kohya_optimizer_group, kohya_saving_group,
                kohya_noise_group, vae_chunk_size, cache_latents,
                cache_text_encoder_outputs, vae_disable_cache,
            ],
        )

        # ── Language save ────────────────────────────────────────────────
        def _save_language(lang: str):
            if lang in SUPPORTED_LANGS:
                set_lang(lang)
                return t("language_saved", lang=lang)
            return ""

        save_lang_btn.click(fn=_save_language, inputs=[language_dd], outputs=[language_status])

        # ── Input groups ─────────────────────────────────────────────────
        adv_inputs = [
            optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
            train_batch_size, gradient_accumulation_steps, max_grad_norm,
            save_every_n_epochs, save_last_n_epochs, mixed_precision,
            gradient_checkpointing, seed, noise_offset, multires_noise_discount,
            timestep_sampling, discrete_flow_shift,
            cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
            num_cpu_threads,
        ]
        basic_inputs = [
            project_name, base_model_dropdown, image_directory, output_directory,
            network_dim, network_alpha, learning_rate, max_train_epochs,
            resolution, repeats, caption_dropout, gpu_dropdown,
        ]
        diffsynth_inputs = [lora_target_modules_in, dataset_repeat_in, max_pixels_in, save_steps_ds_in]
        tb_inputs = [use_tb_chk, tb_logdir_in, tb_port_in]

        # ── Configure Training event ─────────────────────────────────────
        configure_btn.click(
            fn=configure_training,
            inputs=[backend_radio, diffsynth_dir_in] + basic_inputs + adv_inputs + diffsynth_inputs + tb_inputs,
            outputs=[status_box, last_train_cfg, last_dataset_cfg, last_diffsynth_args, last_tb_logdir_state],
        )

        # ── Start Training event ─────────────────────────────────────────
        train_btn.click(
            fn=start_training,
            inputs=[backend_radio, diffsynth_dir_in, custom_config_input, gpu_dropdown, num_cpu_threads, base_model_dropdown, use_tb_chk],
            outputs=[log_box],
        )

        # ── TensorBoard control ──────────────────────────────────────────
        def _start_tb_handler(logdir, port, state_logdir, use_ngrok, ngrok_token):
            # Prefer explicit input, else fall back to last run's logdir
            effective = (logdir or "").strip() or (state_logdir or "")
            # Persist ngrok prefs so the user doesn't need to retype the token
            save_config({
                "ngrok_enable": bool(use_ngrok),
                "ngrok_token": (ngrok_token or "").strip(),
                "tb_port": int(port) if port else 6006,
            })
            return start_tensorboard(effective, port, use_ngrok=bool(use_ngrok), ngrok_token=ngrok_token or "")

        start_tb_btn.click(
            fn=_start_tb_handler,
            inputs=[tb_logdir_in, tb_port_in, last_tb_logdir_state, ngrok_enable_chk, ngrok_token_in],
            outputs=[tb_status_md, tb_iframe],
        )
        stop_tb_btn.click(fn=stop_tensorboard, inputs=[], outputs=[tb_status_md, tb_iframe])

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    share_requested = IS_COLAB or os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    launch_kwargs = {
        "server_name": "0.0.0.0" if share_requested else "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
    }
    if share_requested:
        launch_kwargs["share"] = True
        print(t("info_colab_detected") if IS_COLAB else t("info_gradio_share"))
    demo.launch(**launch_kwargs)
