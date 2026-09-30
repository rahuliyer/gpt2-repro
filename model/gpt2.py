from model.transformer import TransformerBlock
import torch
import torch.nn as nn

from safetensors.torch import save_model, load_model

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.weight_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embedding = nn.Embedding(config.context_len, config.n_embed)

        self.ln1 = nn.LayerNorm(self.config.n_embed)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=False)
        # share weights
        self.lm_head.weight = self.weight_embedding.weight

        self.dropout = nn.Dropout(self.config.dropout)

        self.transformers = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.n_layers)
        ])

    def forward(self, x):
        emb = (self.weight_embedding(x) + 
               self.pos_embedding(torch.arange(x.shape[1], device=x.device)))

        res = self.dropout(emb)

        for t in self.transformers:
            res = t(res)

        res = self.ln1(res)

        return self.lm_head(res)

    @classmethod
    def load_from_checkpoint(cls, config, checkpoint_path):
        model = cls(config)
        load_model(model, checkpoint_path)

        return model

    @classmethod
    def load_from_hf_model(cls, config, hf_model):
        model = cls(config)

        with torch.no_grad():
            # Load the embeddings
            model.weight_embedding.weight.copy_(hf_model.transformer.wte.weight)
            model.pos_embedding.weight.copy_(hf_model.transformer.wpe.weight)

            # copy transformer blocks
            for i in range(config.n_layers):
                model.transformers[i].ln1.weight.copy_(hf_model.transformer.h[i].ln_1.weight)
                model.transformers[i].ln1.bias.copy_(hf_model.transformer.h[i].ln_1.bias)

                c_attn_w = hf_model.transformer.h[i].attn.c_attn.weight
                c_attn_b = hf_model.transformer.h[i].attn.c_attn.bias

                c_attn_w = c_attn_w.view(config.n_embed, 3, config.d_model)
                c_attb_b = c_attn_b.view(3, config.d_model)

                model.transformers[i].Q.weight.copy_(c_attn_w[:, 0, :].T)
                model.transformers[i].Q.bias.copy_(c_attb_b[0])

                model.transformers[i].K.weight.copy_(c_attn_w[:, 1, :].T)
                model.transformers[i].K.bias.copy_(c_attb_b[1])

                model.transformers[i].V.weight.copy_(c_attn_w[:, 2, :].T)
                model.transformers[i].V.bias.copy_(c_attb_b[2])

                model.transformers[i].attn_proj.weight.copy_(hf_model.transformer.h[i].attn.c_proj.weight.T)
                model.transformers[i].attn_proj.bias.copy_(hf_model.transformer.h[i].attn.c_proj.bias)

                model.transformers[i].ln2.weight.copy_(hf_model.transformer.h[i].ln_2.weight)
                model.transformers[i].ln2.bias.copy_(hf_model.transformer.h[i].ln_2.bias)

                model.transformers[i].ffwd[0].weight.copy_(hf_model.transformer.h[i].mlp.c_fc.weight.T)
                model.transformers[i].ffwd[0].bias.copy_(hf_model.transformer.h[i].mlp.c_fc.bias)

                model.transformers[i].ffwd[2].weight.copy_(hf_model.transformer.h[i].mlp.c_proj.weight.T)
                model.transformers[i].ffwd[2].bias.copy_(hf_model.transformer.h[i].mlp.c_proj.bias)

            # copy final layer norm
            model.ln1.weight.copy_(hf_model.transformer.ln_f.weight)
            model.ln1.bias.copy_(hf_model.transformer.ln_f.bias)

        return model
    
    def save(self, checkpoint_path):
        save_model(self, checkpoint_path)

