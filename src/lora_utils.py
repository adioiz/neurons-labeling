"""
LoRA (Low-Rank Adaptation) utilities for SAM2 fine-tuned model loading.

This module contains only the inference-related LoRA code needed to load
fine-tuned model weights. For training, see the sam2-neuron-finetuning repository.
"""

import re
from pathlib import Path
from typing import List, Optional, Set

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank adapter.

    forward(x) = original(x) + scale * dropout(B(A(x)))
    where scale = alpha / rank
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout on the low-rank path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform and B with zeros
        # so the adapter starts as identity (no change to original output)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the original linear
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return original_out + self.scale * lora_out


# Default LoRA target patterns for SAM 2
# These patterns work for both SAM2Train (model.xxx) and video predictor (xxx) structures
SAM2_LORA_TARGETS = [
    # Mask decoder transformer attention layers (with optional model. prefix)
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?sam_mask_decoder\.transformer\.layers\.\d+\.cross_attn_image_to_token\.(q_proj|k_proj|v_proj|out_proj)",
    # Mask decoder final attention
    r"(model\.)?sam_mask_decoder\.transformer\.final_attn_token_to_image\.(q_proj|k_proj|v_proj|out_proj)",
    # Memory attention layers
    r"(model\.)?memory_attention\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?memory_attention\.layers\.\d+\.cross_attn_image\.(q_proj|k_proj|v_proj|out_proj)",
    r"(model\.)?memory_attention\.layers\.\d+\.(linear1|linear2)",
]


def _matches_any_pattern(name: str, patterns: List[str]) -> bool:
    """Check if a module name matches any of the target patterns."""
    return any(re.fullmatch(p, name) for p in patterns)


def apply_lora(
    model: nn.Module,
    target_patterns: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Set[str]:
    """
    Apply LoRA adapters to matching nn.Linear layers in the model.

    Args:
        model: The SAM 2 model (SAM2Base or video predictor).
        target_patterns: Regex patterns for module names to wrap.
                        Defaults to SAM2_LORA_TARGETS.
        rank: LoRA rank (low-rank dimension).
        alpha: LoRA scaling factor.
        dropout: Dropout on the LoRA path.

    Returns:
        Set of module names that were wrapped with LoRA.
    """
    if target_patterns is None:
        target_patterns = SAM2_LORA_TARGETS

    wrapped_names = set()

    # Build a flat dict of all named modules for parent lookup
    named_modules = dict(model.named_modules())

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_any_pattern(name, target_patterns):
            continue

        # Split name into parent path and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = named_modules[parent_name]
        else:
            parent = model
            attr_name = parts[0]

        # Replace the Linear with LoRALinear
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora_layer)

        wrapped_names.add(name)

    return wrapped_names


def _normalize_key(key: str) -> str:
    """Remove 'model.' prefix from key for consistent storage."""
    if key.startswith("model."):
        return key[6:]
    return key


def load_lora_weights(model: nn.Module, path: str, device: str = "cpu"):
    """
    Load LoRA weights into a model that already has LoRA applied.

    Args:
        model: Model with LoRA adapters already applied via apply_lora().
        path: Path to the saved LoRA weights file.
        device: Device to load weights onto.

    Handles both old checkpoints (with 'model.' prefix) and new checkpoints
    (normalized without prefix) for compatibility.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    lora_state = checkpoint["lora_state_dict"]
    other_state = checkpoint.get("other_state_dict", {})

    # Build a mapping from LoRA module names to their modules
    # This ensures we move the actual module objects to the device
    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules[name] = module
            # Move entire LoRA module to device
            module.to(device)

    # Get current model state after moving modules
    model_state = model.state_dict()

    # Build a normalized key mapping for the model state
    # This maps both "xxx" and "model.xxx" to the actual key in model_state
    model_key_map = {}
    for key in model_state.keys():
        model_key_map[key] = key
        # Also map normalized version
        normalized = _normalize_key(key)
        if normalized != key:
            model_key_map[normalized] = key

    # Try to match keys, handling potential "model." prefix mismatch
    loaded_count = 0
    for ckpt_key, value in lora_state.items():
        # Normalize the checkpoint key
        normalized_ckpt_key = _normalize_key(ckpt_key)

        # Try to find matching model key
        model_key = None
        if ckpt_key in model_key_map:
            model_key = model_key_map[ckpt_key]
        elif normalized_ckpt_key in model_key_map:
            model_key = model_key_map[normalized_ckpt_key]

        if model_key is not None:
            model_state[model_key].copy_(value.to(device))
            loaded_count += 1
        else:
            # Only warn for LoRA weights
            if "lora_" in ckpt_key:
                print(f"Warning: LoRA key {ckpt_key} not found in model")

    # Load other trainable weights with same handling
    for ckpt_key, value in other_state.items():
        normalized_ckpt_key = _normalize_key(ckpt_key)

        model_key = None
        if ckpt_key in model_key_map:
            model_key = model_key_map[ckpt_key]
        elif normalized_ckpt_key in model_key_map:
            model_key = model_key_map[normalized_ckpt_key]

        if model_key is not None:
            model_state[model_key].copy_(value.to(device))
        # Don't warn for other state - may have different structure

    print(f"Loaded LoRA weights from {path} ({loaded_count} LoRA tensors)")
