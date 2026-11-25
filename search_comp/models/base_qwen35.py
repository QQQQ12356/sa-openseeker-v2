"""Load the original Qwen3.5 text backbone for independent baseline inference."""
from pathlib import Path
import sys


def load_base_qwen35(model_path):
    """Load an original Qwen3.5 checkpoint, without PEFT or Plain SFT loaders."""
    from transformers import AutoConfig
    from ..milestones.qwen35_text import load_text_causal_model

    path = Path(model_path).expanduser()
    if (path / 'adapter_config.json').is_file():
        raise ValueError('Qwen3.5 baseline requires original full weights, not an adapter')
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.model_type != 'qwen3_5':
        raise ValueError(f'baseline requires the original Qwen3.5 multimodal checkpoint, got {config.model_type}')
    if getattr(config, 'enable_beacon', False) or getattr(getattr(config, 'text_config', None), 'enable_beacon', False):
        raise ValueError('Beacon checkpoint cannot be used as an original Qwen3.5 baseline')
    try:
        return load_text_causal_model(model_path)
    except ImportError as error:
        if "causal_conv1d_cuda" not in str(error) or "undefined symbol" not in str(error):
            raise
        raise RuntimeError(
            f"causal-conv1d is incompatible with the PyTorch loaded by {sys.executable}. "
            "Rebuild causal-conv1d from source in this Linux environment before rerunning. "
            "See docs/10_environment.md, Qwen3.5 causal-conv1d troubleshooting."
        ) from error
