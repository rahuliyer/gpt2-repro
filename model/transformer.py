import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        assert config.d_model % config.n_heads == 0

        self.Q = nn.Linear(config.n_embed, config.d_model)
        self.K = nn.Linear(config.n_embed, config.d_model)
        self.V = nn.Linear(config.n_embed, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

        self.attn_proj = nn.Linear(config.d_model, config.n_embed)

        self.ffwd = nn.Sequential(
                nn.Linear(config.n_embed, 4 * config.n_embed),
                nn.GELU(approximate="tanh"),
                nn.Linear(4 * config.n_embed, config.n_embed)
        )

    def forward(self, x):
        res = self.ln1(x)

        q = self.Q(res) # B x T x d_model
        k = self.K(res) # B x T x d_model
        v = self.V(res) # B x T x d_model

        q = q.view(x.shape[0], 
                   x.shape[1], 
                   self.config.n_heads, 
                   self.config.d_model // self.config.n_heads) # B x T x H x head_size
        k = k.view(x.shape[0], 
                   x.shape[1], 
                   self.config.n_heads, 
                   self.config.d_model // self.config.n_heads) # B x T x H x head_size
        v = v.view(x.shape[0], 
                   x.shape[1], 
                   self.config.n_heads, 
                   self.config.d_model // self.config.n_heads) # B x T x H x head_size

        q = q.transpose(1, 2) # B x H x T x head_size
        k = k.transpose(1, 2) # B x H x T x head_size
        v = v.transpose(1, 2) # B x H x T x head_size

        #attn_matrix = (q @ k.transpose(-1, -2) / math.sqrt(self.config.d_model // self.config.n_heads)) # B x H x T x T
        #mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device)) # B x H x T x T
        #attn_mask = attn_matrix.masked_fill(mask == 0, float('-inf')) # B x H x T x T
        #attn_weights = torch.softmax(attn_mask, dim=-1)
        #attn_weights = self.dropout(attn_weights)

        #res = attn_weights @ v # B x H x T x head_size

        # Use scaled dot product attention
        res = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=True,
        )

        res = res.transpose(1, 2) # B x T x H x head_size
        res = res.reshape(x.shape[0], x.shape[1], self.config.d_model) # B x T x d_model
        res = self.attn_proj(res) # B x T x n_embed

        res = self.dropout(res)
 
        res = res + x

        x = res
        res = self.ln2(res)
        res = self.ffwd(res)
        res = self.dropout(res)
        res = res + x

        return res
