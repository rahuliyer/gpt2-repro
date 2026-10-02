"""Turn a pretrained GPT-2 into a sequence classifier."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from model import GPT2, GPT2SmallConfig

HEADS = ("mlp", "linear")
UNFREEZE = ("none", "last_block", "all")


def build_head(n_embed, head, num_classes):
    if head == "mlp":
        return nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, num_classes),
        )
    if head == "linear":
        return nn.Linear(n_embed, num_classes)
    raise ValueError(f"head must be one of {HEADS}, got {head!r}")


def build_classifier(backbone, head="mlp", unfreeze="none", num_classes=4):
    """Swap the LM head of `backbone` for a classification head.

    The new head always trains. `unfreeze` picks how much of the backbone
    trains with it: "none", "last_block" (the final transformer block and the
    layer norm after it) or "all".
    """
    if unfreeze not in UNFREEZE:
        raise ValueError(f"unfreeze must be one of {UNFREEZE}, got {unfreeze!r}")

    for param in backbone.parameters():
        param.requires_grad = unfreeze == "all"

    if unfreeze == "last_block":
        for module in (backbone.transformers[-1], backbone.ln1):
            for param in module.parameters():
                param.requires_grad = True

    # Replacing the head also drops its weight tying with the token embedding.
    backbone.lm_head = build_head(backbone.config.n_embed, head, num_classes)

    return backbone


def load_pretrained_classifier(head="mlp", unfreeze="none", num_classes=4):
    """Build a classifier on top of the released GPT-2 small weights."""
    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    backbone = GPT2.load_from_hf_model(GPT2SmallConfig(dropout=0.0), hf_model)

    return build_classifier(backbone, head, unfreeze, num_classes)


def get_last_logits(logits, lengths):
    """Pick each sequence's logits at its last real token, skipping the padding."""
    return logits[torch.arange(logits.shape[0], device=logits.device), lengths - 1]
