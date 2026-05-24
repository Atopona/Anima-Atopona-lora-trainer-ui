#!/usr/bin/env bash
# =============================================================================
# run_linux.sh — Activate venv and launch Anima LoRA Trainer
#
# Usage:
#   bash run_linux.sh             # local: http://127.0.0.1:7860
#   bash run_linux.sh --share     # force Gradio share=True (gradio.live tunnel)
#
# Auto-detects Colab (COLAB_GPU / COLAB_RELEASE_TAG) and enables share=True
# without needing any flag.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IS_COLAB=0
if [ -n "${COLAB_GPU:-}" ] || [ -n "${COLAB_RELEASE_TAG:-}" ]; then
    IS_COLAB=1
fi

# --- Activate venv (skip on Colab, which uses system Python) -----------------
if [ "$IS_COLAB" = "0" ]; then
    if [ ! -d ".venv" ]; then
        echo "ERROR: .venv not found. Run setup_for_linux.sh first."
        exit 1
    fi
    source .venv/bin/activate
fi

# --- Share-mode detection ----------------------------------------------------
SHARE_MODE=$IS_COLAB
for arg in "$@"; do
    case "$arg" in
        --share|--colab|--tunnel)
            SHARE_MODE=1
            ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

if [ "$SHARE_MODE" = "1" ]; then
    export GRADIO_SHARE=1

    # Auto-install pyngrok so the in-UI TensorBoard tunnel works out of the box.
    if ! python -c "import pyngrok" >/dev/null 2>&1; then
        echo "Installing pyngrok for TensorBoard tunneling..."
        pip install --quiet pyngrok
    fi
    # Ensure tensorboard is available too (in case the user skipped setup).
    if ! python -c "import tensorboard" >/dev/null 2>&1; then
        echo "Installing tensorboard..."
        pip install --quiet "tensorboard>=2.10"
    fi

    echo "============================================================"
    if [ "$IS_COLAB" = "1" ]; then
        echo "  Anima LoRA Trainer — Colab detected, SHARE MODE on"
    else
        echo "  Anima LoRA Trainer — SHARE MODE (--share)"
    fi
    echo "  • Gradio UI: a *.gradio.live URL will print below."
    echo "  • TensorBoard: open the TensorBoard tab, enable ngrok,"
    echo "    paste your token from https://dashboard.ngrok.com/"
    echo "    get-started/your-authtoken, then click Start TensorBoard."
    echo "============================================================"
else
    echo "Starting Anima LoRA Trainer at http://127.0.0.1:7860 ..."
fi

python app.py
