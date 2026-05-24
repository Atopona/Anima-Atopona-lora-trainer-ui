"""
Lightweight i18n for Anima LoRA Trainer.

Usage:
    from i18n import t, get_lang, set_lang, SUPPORTED_LANGS
    label = t("project_name")            # auto-picks current language
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"

SUPPORTED_LANGS = ["en", "zh"]
DEFAULT_LANG = "en"

_LANG_CACHE: dict = {"value": None}


TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Page header
        "app_title": "Anima LoRA Trainer",
        "header_markdown": (
            "# 🍋 Citron's Anima LoRA Trainer\n\n"
            "Super Simple Gradio UI for training LoRA adapters on the "
            "<a href=\"https://huggingface.co/circlestone-labs/Anima\" target=\"_blank\" rel=\"noopener noreferrer\">Anima</a> "
            "diffusion model. Supports both "
            "<a href=\"https://github.com/kohya-ss/sd-scripts\" target=\"_blank\" rel=\"noopener noreferrer\">kohya-ss/sd-scripts</a> "
            "and <a href=\"https://github.com/modelscope/DiffSynth-Studio\" target=\"_blank\" rel=\"noopener noreferrer\">DiffSynth-Studio</a> backends.\n\n"
            "🚀 Runs on ~6 GB VRAM with default settings.\n\n"
            "Created by <a href=\"https://x.com/Citron_Legacy\" target=\"_blank\" rel=\"noopener noreferrer\">Citron Legacy</a>. "
            "Source: <a href=\"https://github.com/citronlegacy/citron-anima-lora-trainer-ui\" target=\"_blank\" rel=\"noopener noreferrer\">GitHub</a>."
        ),
        # Top bar
        "language": "Language",
        "language_info": "Switch UI language. Saved to config.json — refresh the page after changing.",
        "language_saved": "✓ Language saved as '{lang}'. Please refresh the browser page to apply.",
        "backend": "Training Backend",
        "backend_info": "kohya-ss/sd-scripts (TOML configs, full optimizer choices) or DiffSynth-Studio (CLI args, simpler).",
        # Tabs
        "tab_training": "Training",
        "tab_advanced": "Advanced Settings",
        "tab_tensorboard": "TensorBoard",
        # Sections
        "section_project_paths": "Project & Paths",
        "section_network": "Network",
        "section_dataset": "Dataset",
        "section_config_training": "Config & Training",
        "section_optimizer": "Optimizer & Scheduler",
        "section_batch": "Batch & Gradient",
        "section_saving": "Saving",
        "section_precision": "Precision & Memory",
        "section_noise": "Noise & Flow",
        "section_misc": "Misc",
        "section_diffsynth": "DiffSynth-Specific",
        "section_tb_settings": "TensorBoard Settings",
        # Basic fields
        "project_name": "Project Name",
        "gpu": "GPU",
        "base_model": "Base Model",
        "base_model_info": "Selected base model auto-downloads when you click Start Training (first run only).",
        "image_directory": "Image Directory (flat folder with images + .txt captions)",
        "output_directory": "Output Directory (where the trained LoRA is saved)",
        "network_dim": "Network Dim",
        "network_alpha": "Network Alpha",
        "learning_rate": "Learning Rate",
        "max_epochs": "Max Epochs",
        "resolution": "Resolution (px)",
        "repeats": "Repeats",
        "caption_dropout": "Caption Dropout",
        # Buttons
        "btn_configure": "⚙️ Configure Training",
        "btn_start": "🚀 Start Training",
        "btn_stop_training": "🛑 Stop Training",
        "btn_start_tb": "📊 Start TensorBoard",
        "btn_stop_tb": "⏹ Stop TensorBoard",
        "btn_open_tb": "Open in new tab",
        "btn_open_local": "Local URL",
        "btn_save_language": "Save Language",
        # Status boxes
        "status_label": "Configuration Status",
        "log_label": "Training Log",
        "override_config_label": "Override Training Config Path (optional — leave blank to use last generated)",
        # Advanced
        "optimizer": "Optimizer",
        "lr_scheduler": "LR Scheduler",
        "lr_scheduler_cycles": "LR Scheduler Num Cycles",
        "lr_warmup_steps": "LR Warmup Steps",
        "train_batch_size": "Train Batch Size",
        "grad_accum_steps": "Gradient Accumulation Steps",
        "max_grad_norm": "Max Grad Norm",
        "save_every_n_epochs": "Save Every N Epochs",
        "save_last_n": "Keep Last N Checkpoints",
        "mixed_precision": "Mixed Precision",
        "vae_chunk_size": "VAE Chunk Size",
        "gradient_checkpointing": "Gradient Checkpointing",
        "cache_latents": "Cache Latents",
        "cache_text_encoder": "Cache Text Encoder Outputs",
        "vae_disable_cache": "VAE Disable Cache",
        "noise_offset": "Noise Offset",
        "multires_noise_discount": "Multires Noise Discount",
        "timestep_sampling": "Timestep Sampling",
        "discrete_flow_shift": "Discrete Flow Shift",
        "seed": "Seed",
        "cpu_threads": "CPU Threads Per Process",
        # DiffSynth-specific
        "lora_target_modules": "LoRA Target Modules",
        "lora_target_modules_info": "Leave blank for DiffSynth's Anima default. Custom comma-separated module names are advanced use only.",
        "dataset_repeat": "Dataset Repeat (per epoch)",
        "max_pixels": "Max Pixels (dynamic resolution)",
        "save_steps_ds": "Save Every N Steps (DiffSynth)",
        "save_steps_ds_info": "Leave 0 to save once per epoch.",
        "diffsynth_dir": "DiffSynth-Studio Directory",
        "diffsynth_dir_info": "Path to a DiffSynth-Studio clone. Leave blank to auto-clone/install ./DiffSynth-Studio under the project root.",
        "diffsynth_lr_note": "ℹ DiffSynth currently uses AdamW + ConstantLR — optimizer/scheduler controls don't apply.",
        # TensorBoard tab
        "tb_use": "Enable TensorBoard logging",
        "tb_use_info": "Adds --log_with tensorboard to kohya, or writes losses to TF events for DiffSynth.",
        "tb_logdir": "Log Directory",
        "tb_logdir_info": "Where TF event files live. Auto-defaults to logs/tb/<project>_<timestamp>.",
        "tb_port": "TensorBoard Port",
        "tb_status": "TensorBoard Status",
        "tb_status_stopped": "TensorBoard is not running.",
        "tb_status_running": "TensorBoard running at {url}",
        "tb_status_failed": "❌ Failed to start TensorBoard: {err}",
        "tb_status_already": "TensorBoard already running at {url}.",
        "tb_status_no_logdir": "❌ Log directory does not exist yet — start training first or pick an existing path.",
        "tb_view_placeholder": "TensorBoard panel will appear here after starting.",
        # Sharing / ngrok / Colab
        "section_sharing": "Sharing / Tunneling (for Colab or remote use)",
        "ngrok_enable": "Use ngrok tunnel for TensorBoard (required on Colab)",
        "ngrok_enable_info": "Exposes the local TB port via a public *.ngrok-free.app URL so the iframe is reachable from a remote browser (e.g. Colab).",
        "ngrok_token": "ngrok Auth Token",
        "ngrok_token_info": "Get yours at https://dashboard.ngrok.com/get-started/your-authtoken. Saved to config.json — keep that file out of source control.",
        "ngrok_status_no_token": "❌ ngrok requires an auth token. Paste yours into the field above (or set $NGROK_AUTHTOKEN).",
        "ngrok_status_no_pyngrok": "⚠ pyngrok is not installed. Run: pip install pyngrok",
        "ngrok_status_tunnel_open": "🌐 Public URL: {url}",
        "ngrok_status_tunnel_failed": "❌ ngrok tunnel failed: {err}",
        "info_colab_detected": "🌐 Colab environment detected — Gradio will launch with share=True.",
        "info_gradio_share": "Gradio share enabled via $GRADIO_SHARE.",
        # Configure_training messages
        "err_project_empty": "❌ Project name cannot be empty.",
        "err_image_dir_empty": "❌ Image directory cannot be empty.",
        "err_output_dir_empty": "❌ Output directory cannot be empty.",
        "info_project": "Project:          {name}",
        "info_image_dir": "Image directory:  {dir}",
        "info_output_dir": "Output directory: {dir}",
        "info_backend": "Backend:          {backend}",
        "info_images_found": "Images found:     {n}",
        "info_missing_captions": "⚠ Missing captions ({n}):",
        "info_more": "    … and {n} more",
        "info_all_have_captions": "✓ All images have caption files.",
        "info_no_images": "❌ Cannot configure — no images found.",
        "info_step_header": "── Step Estimate ─────────────────────────────────────",
        "info_step_per_epoch": "  Steps per epoch: {n}  ({imgs} imgs × {repeats} repeats)",
        "info_step_total": "  Total steps:     {n}  ({spe} × {ep} epochs)",
        "info_step_diffsynth_note": "  Note: DiffSynth progress uses repeated dataset samples, not Kohya batch/accumulation optimizer steps.",
        "info_step_footer": "──────────────────────────────────────────────────────",
        "info_checking_models": "Checking models...",
        "info_will_download": "      (will auto-download when training starts)",
        "info_missing_models": "❌ Missing models: {list}",
        "info_run_setup": "Run setup_for_linux.sh / setup_for_windows.bat to download them.",
        "info_generating_toml": "Generating TOML configs...",
        "info_generating_metadata": "Generating DiffSynth metadata.csv and CLI args...",
        "info_metadata_written": "  ✓ metadata.csv: {path} ({n} rows)",
        "info_args_written": "  ✓ DiffSynth args: {path}",
        "info_train_cfg_written": "  ✓ Training config: {path}",
        "info_dataset_cfg_written": "  ✓ Dataset  config: {path}",
        "info_ready": "✓ Configuration complete — ready to train.",
        "err_generate_failed": "❌ Failed to generate configs: {err}",
        "err_diffsynth_missing": "❌ DiffSynth-Studio is not ready at: {path}\nLeave the DiffSynth directory blank, or choose an empty directory / valid DiffSynth-Studio clone.",
        "info_diffsynth_will_install": "ℹ DiffSynth-Studio is not ready at {path}. It will be cloned/updated and installed automatically when training starts.",
        "info_diffsynth_check": "🔎 Checking DiffSynth-Studio install for the current Python interpreter...",
        "err_diffsynth_install_failed": "❌ DiffSynth-Studio install failed: {err}",
        # start_training messages
        "info_using_gpu": "Using GPU index: {idx}",
        "info_command": "Command: {cmd}",
        "info_downloading_model": "⏳ Downloading base model '{name}'...",
        "info_download_destination": "   Destination: {path}",
        "info_download_done": "✓ Base model downloaded successfully.",
        "err_no_train_cfg": "❌ No training config found. Run 'Configure Training' first, or provide a config path.",
        "err_train_cfg_not_found": "❌ Training config not found: {path}",
        "err_no_dataset_cfg": "❌ No dataset config found. Run 'Configure Training' first.",
        "err_dataset_cfg_not_found": "❌ Dataset config not found: {path}",
        "err_train_script_missing": "❌ Training script not found: {path}\nRun setup_for_linux.sh / setup_for_windows.bat first.",
        "err_accelerate_missing": "❌ 'accelerate' not found. Make sure the venv is activated and accelerate is installed.",
        "err_unknown_base_model": "❌ Unknown base model: {name}",
        "err_wget_missing": "❌ 'wget' not found. Install wget and try again.",
        "err_download_failed": "❌ Download failed (exit code {code})",
        "info_train_done": "\n✓ Training completed successfully!\nLoRA saved to: {output}\nLog saved to: {log}",
        "info_train_failed": "\n✗ Training failed (exit code: {code})\nLog saved to: {log}",
        "info_oom_hint": "\n💡 OOM detected in kernel log. Try: network_dim=8 and/or resolution=512",
        "info_tb_enabled": "📊 TensorBoard logging enabled. Logs → {dir}",
    },

    "zh": {
        # Page header
        "app_title": "Anima LoRA 训练器",
        "header_markdown": (
            "# 🍋 Citron 的 Anima LoRA 训练器\n\n"
            "用于在 <a href=\"https://huggingface.co/circlestone-labs/Anima\" target=\"_blank\" rel=\"noopener noreferrer\">Anima</a> "
            "扩散模型上训练 LoRA 适配器的极简 Gradio UI。同时支持 "
            "<a href=\"https://github.com/kohya-ss/sd-scripts\" target=\"_blank\" rel=\"noopener noreferrer\">kohya-ss/sd-scripts</a> "
            "与 <a href=\"https://github.com/modelscope/DiffSynth-Studio\" target=\"_blank\" rel=\"noopener noreferrer\">DiffSynth-Studio</a> 两种后端。\n\n"
            "🚀 默认设置下显存占用约 6 GB。\n\n"
            "作者：<a href=\"https://x.com/Citron_Legacy\" target=\"_blank\" rel=\"noopener noreferrer\">Citron Legacy</a>。 "
            "源码：<a href=\"https://github.com/citronlegacy/citron-anima-lora-trainer-ui\" target=\"_blank\" rel=\"noopener noreferrer\">GitHub</a>。"
        ),
        # Top bar
        "language": "界面语言",
        "language_info": "切换 UI 语言。保存到 config.json — 切换后请刷新页面。",
        "language_saved": "✓ 语言已保存为 '{lang}'。请刷新浏览器页面以应用。",
        "backend": "训练后端",
        "backend_info": "kohya-ss/sd-scripts（TOML 配置，优化器选项丰富）或 DiffSynth-Studio（命令行参数，更简洁）。",
        # Tabs
        "tab_training": "训练",
        "tab_advanced": "高级设置",
        "tab_tensorboard": "TensorBoard",
        # Sections
        "section_project_paths": "项目与路径",
        "section_network": "网络结构",
        "section_dataset": "数据集",
        "section_config_training": "配置与训练",
        "section_optimizer": "优化器与调度器",
        "section_batch": "批量与梯度",
        "section_saving": "保存设置",
        "section_precision": "精度与显存",
        "section_noise": "噪声与流匹配",
        "section_misc": "其他",
        "section_diffsynth": "DiffSynth 专属参数",
        "section_tb_settings": "TensorBoard 设置",
        # Basic
        "project_name": "项目名称",
        "gpu": "GPU 设备",
        "base_model": "基础模型",
        "base_model_info": "首次点击「开始训练」时会自动下载所选基础模型（仅首次）。",
        "image_directory": "图像目录（扁平文件夹，每张图配对 .txt 标注）",
        "output_directory": "输出目录（训练好的 LoRA 保存位置）",
        "network_dim": "网络维度 Dim",
        "network_alpha": "网络 Alpha",
        "learning_rate": "学习率",
        "max_epochs": "最大轮数 Epochs",
        "resolution": "分辨率（像素）",
        "repeats": "重复次数 Repeats",
        "caption_dropout": "标注丢弃率",
        # Buttons
        "btn_configure": "⚙️ 生成训练配置",
        "btn_start": "🚀 开始训练",
        "btn_stop_training": "🛑 停止训练",
        "btn_start_tb": "📊 启动 TensorBoard",
        "btn_stop_tb": "⏹ 停止 TensorBoard",
        "btn_open_tb": "在新标签页打开",
        "btn_open_local": "本地 URL",
        "btn_save_language": "保存语言",
        # Status
        "status_label": "配置状态",
        "log_label": "训练日志",
        "override_config_label": "覆盖训练配置路径（可选 — 留空使用最近生成的配置）",
        # Advanced
        "optimizer": "优化器",
        "lr_scheduler": "学习率调度器",
        "lr_scheduler_cycles": "调度器周期数",
        "lr_warmup_steps": "学习率预热步数",
        "train_batch_size": "训练批量大小",
        "grad_accum_steps": "梯度累积步数",
        "max_grad_norm": "梯度裁剪阈值",
        "save_every_n_epochs": "每 N 轮保存一次",
        "save_last_n": "保留最近 N 个检查点",
        "mixed_precision": "混合精度",
        "vae_chunk_size": "VAE 分块大小",
        "gradient_checkpointing": "启用梯度检查点",
        "cache_latents": "缓存潜变量",
        "cache_text_encoder": "缓存文本编码输出",
        "vae_disable_cache": "禁用 VAE 缓存",
        "noise_offset": "噪声偏移 Noise Offset",
        "multires_noise_discount": "多分辨率噪声折扣",
        "timestep_sampling": "时间步采样",
        "discrete_flow_shift": "离散流偏移",
        "seed": "随机种子",
        "cpu_threads": "每进程 CPU 线程数",
        # DiffSynth
        "lora_target_modules": "LoRA 目标模块",
        "lora_target_modules_info": "留空使用 DiffSynth 的 Anima 默认值。仅高级自定义时填写逗号分隔的模块名。",
        "dataset_repeat": "数据集每轮重复次数",
        "max_pixels": "最大像素数（动态分辨率）",
        "save_steps_ds": "每 N 步保存一次（DiffSynth）",
        "save_steps_ds_info": "填 0 表示仅按轮保存。",
        "diffsynth_dir": "DiffSynth-Studio 目录",
        "diffsynth_dir_info": "DiffSynth-Studio 克隆路径。留空则自动克隆/安装项目根下的 ./DiffSynth-Studio。",
        "diffsynth_lr_note": "ℹ DiffSynth 目前固定使用 AdamW + ConstantLR — 优化器/调度器控件不生效。",
        # TensorBoard
        "tb_use": "启用 TensorBoard 日志",
        "tb_use_info": "kohya 端通过 --log_with tensorboard 接入；DiffSynth 端由本程序解析 loss 写入 TF event。",
        "tb_logdir": "日志目录",
        "tb_logdir_info": "TF event 文件位置。默认自动定为 logs/tb/<项目名>_<时间戳>。",
        "tb_port": "TensorBoard 端口",
        "tb_status": "TensorBoard 状态",
        "tb_status_stopped": "TensorBoard 未运行。",
        "tb_status_running": "TensorBoard 已运行：{url}",
        "tb_status_failed": "❌ TensorBoard 启动失败：{err}",
        "tb_status_already": "TensorBoard 已在运行：{url}",
        "tb_status_no_logdir": "❌ 日志目录尚不存在 — 请先开始训练或选择已有路径。",
        "tb_view_placeholder": "启动 TensorBoard 后将在此显示。",
        # Sharing / ngrok / Colab
        "section_sharing": "公开访问 / 内网穿透（Colab 或远程使用）",
        "ngrok_enable": "为 TensorBoard 启用 ngrok 隧道（Colab 必须开启）",
        "ngrok_enable_info": "通过 *.ngrok-free.app 公开 URL 包装本地 TB 端口，使远程浏览器（如 Colab）能从 iframe 中访问。",
        "ngrok_token": "ngrok 鉴权 Token",
        "ngrok_token_info": "在 https://dashboard.ngrok.com/get-started/your-authtoken 获取。保存到 config.json — 请勿将该文件提交到版本库。",
        "ngrok_status_no_token": "❌ ngrok 需要 auth token。请在上方粘贴（或设置环境变量 $NGROK_AUTHTOKEN）。",
        "ngrok_status_no_pyngrok": "⚠ 未安装 pyngrok。请运行：pip install pyngrok",
        "ngrok_status_tunnel_open": "🌐 公开 URL：{url}",
        "ngrok_status_tunnel_failed": "❌ ngrok 隧道建立失败：{err}",
        "info_colab_detected": "🌐 检测到 Colab 环境 — Gradio 将自动以 share=True 启动。",
        "info_gradio_share": "已通过环境变量 $GRADIO_SHARE 启用 Gradio 共享。",
        # Configure messages
        "err_project_empty": "❌ 项目名称不能为空。",
        "err_image_dir_empty": "❌ 图像目录不能为空。",
        "err_output_dir_empty": "❌ 输出目录不能为空。",
        "info_project": "项目：            {name}",
        "info_image_dir": "图像目录：        {dir}",
        "info_output_dir": "输出目录：        {dir}",
        "info_backend": "后端：            {backend}",
        "info_images_found": "找到图片数量：    {n}",
        "info_missing_captions": "⚠ 缺少标注的图片（{n} 张）：",
        "info_more": "    … 还有 {n} 张",
        "info_all_have_captions": "✓ 所有图片均已配对 .txt 标注。",
        "info_no_images": "❌ 无法配置 — 未发现任何图片。",
        "info_step_header": "── 训练步数估算 ─────────────────────────────────────",
        "info_step_per_epoch": "  每轮步数： {n}  （{imgs} 张图 × {repeats} 次重复）",
        "info_step_total": "  总步数：   {n}  （{spe} × {ep} 轮）",
        "info_step_diffsynth_note": "  注：DiffSynth 进度条按重复后的样本数显示，不按 Kohya 的 batch/梯度累积优化步数口径显示。",
        "info_step_footer": "──────────────────────────────────────────────────────",
        "info_checking_models": "检查模型文件...",
        "info_will_download": "      （将在开始训练时自动下载）",
        "info_missing_models": "❌ 缺失模型：{list}",
        "info_run_setup": "请运行 setup_for_linux.sh / setup_for_windows.bat 下载它们。",
        "info_generating_toml": "正在生成 TOML 配置...",
        "info_generating_metadata": "正在生成 DiffSynth 的 metadata.csv 与命令行参数...",
        "info_metadata_written": "  ✓ metadata.csv：{path}（{n} 行）",
        "info_args_written": "  ✓ DiffSynth 参数：{path}",
        "info_train_cfg_written": "  ✓ 训练配置：{path}",
        "info_dataset_cfg_written": "  ✓ 数据集配置：{path}",
        "info_ready": "✓ 配置完成 — 可以开始训练。",
        "err_generate_failed": "❌ 生成配置失败：{err}",
        "err_diffsynth_missing": "❌ DiffSynth-Studio 未就绪：{path}\n请留空 DiffSynth 目录，或选择空目录 / 有效的 DiffSynth-Studio 克隆。",
        "info_diffsynth_will_install": "ℹ 路径 {path} 下的 DiffSynth-Studio 未就绪。开始训练时将自动克隆/更新并安装。",
        "info_diffsynth_check": "🔎 正在检查当前 Python 解释器中的 DiffSynth-Studio 安装状态...",
        "err_diffsynth_install_failed": "❌ DiffSynth-Studio 安装失败：{err}",
        # start_training
        "info_using_gpu": "使用 GPU：{idx}",
        "info_command": "命令：{cmd}",
        "info_downloading_model": "⏳ 正在下载基础模型 '{name}'...",
        "info_download_destination": "   目标位置：{path}",
        "info_download_done": "✓ 基础模型下载完成。",
        "err_no_train_cfg": "❌ 未找到训练配置。请先运行「生成训练配置」或提供配置路径。",
        "err_train_cfg_not_found": "❌ 训练配置不存在：{path}",
        "err_no_dataset_cfg": "❌ 未找到数据集配置。请先运行「生成训练配置」。",
        "err_dataset_cfg_not_found": "❌ 数据集配置不存在：{path}",
        "err_train_script_missing": "❌ 训练脚本不存在：{path}\n请先运行 setup_for_linux.sh / setup_for_windows.bat。",
        "err_accelerate_missing": "❌ 未找到 'accelerate'。请确认虚拟环境已激活且 accelerate 已安装。",
        "err_unknown_base_model": "❌ 未知的基础模型：{name}",
        "err_wget_missing": "❌ 未找到 'wget'。请先安装 wget 后再试。",
        "err_download_failed": "❌ 下载失败（退出码 {code}）",
        "info_train_done": "\n✓ 训练成功完成！\nLoRA 已保存至：{output}\n日志已保存至：{log}",
        "info_train_failed": "\n✗ 训练失败（退出码：{code}）\n日志已保存至：{log}",
        "info_oom_hint": "\n💡 内核日志检测到 OOM。建议：network_dim=8 和/或 resolution=512。",
        "info_tb_enabled": "📊 已启用 TensorBoard 日志，输出至 → {dir}",
    },
}


def _read_lang_from_config() -> str:
    if not CONFIG_FILE.exists():
        return DEFAULT_LANG
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        lang = cfg.get("language", DEFAULT_LANG)
        return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG
    except Exception:
        return DEFAULT_LANG


def get_lang() -> str:
    if _LANG_CACHE["value"] is None:
        _LANG_CACHE["value"] = _read_lang_from_config()
    return _LANG_CACHE["value"]


def set_lang(lang: str) -> None:
    """Update language in config.json and clear cache."""
    if lang not in SUPPORTED_LANGS:
        return
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    cfg["language"] = lang
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    _LANG_CACHE["value"] = lang


def t(key: str, **fmt) -> str:
    """Translate `key` to current language, with optional .format() kwargs."""
    lang = get_lang()
    text = TRANSLATIONS.get(lang, {}).get(key)
    if text is None:
        text = TRANSLATIONS[DEFAULT_LANG].get(key, key)
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError):
            return text
    return text
