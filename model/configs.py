from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    n_embed: int
    context_len: int
    dropout: float

@dataclass
class GPT2SmallConfig(GPTConfig):
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    n_embed: int = 768
    context_len: int = 1024
    dropout: float = 0.0

@dataclass
class GPT2MediumConfig(GPTConfig):
    vocab_size: int = 50257
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 24
    n_embed: int = 1024
    context_len: int = 1024
    dropout: float = 0.0

@dataclass
class GPT2LargeConfig(GPTConfig):
    vocab_size: int = 50257
    d_model: int = 1280
    n_heads: int = 20
    n_layers: int = 36
    n_embed: int = 1280
    context_len: int = 1024
    dropout: float = 0.0


GPTSmallConfig = GPT2SmallConfig
