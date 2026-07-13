"""Small neural-network building blocks used by Nano-Megatron."""

from nano_megatron.nn.activation_checkpoint import (
    ActivationCheckpoint,
    activation_checkpoint,
)
from nano_megatron.nn.dropout import activation_rng_context, parallel_dropout
from nano_megatron.nn.norms import RMSNorm
from nano_megatron.nn.rotary import RotaryEmbedding, apply_rotary_pos_emb

__all__ = [
    "ActivationCheckpoint",
    "RMSNorm",
    "RotaryEmbedding",
    "activation_checkpoint",
    "activation_rng_context",
    "apply_rotary_pos_emb",
    "parallel_dropout",
]
