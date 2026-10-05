import torch
import tiktoken

@torch.no_grad()
def generate_text(
    model, tokenizer, prompt, max_length=50, stop_token_id=None, temperature=1.0, top_k=5
):
    """Sample a continuation of `prompt` and decode the prompt plus continuation.

    `prompt` is a string or a list of token ids (for prompts holding special
    tokens). Sampling stops at `stop_token_id`, which defaults to the
    end-of-text token.

    Each step divides the logits by `temperature` (below 1 sharpens the
    distribution, above 1 flattens it) and samples from only the `top_k` most
    likely tokens. `top_k=None` samples from the full distribution.
    """
    model.eval()
    if stop_token_id is None:
        stop_token_id = tokenizer.eot_token
    if isinstance(prompt, str):
        prompt = tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
    input_ids = torch.tensor(prompt).unsqueeze(0).to(next(model.parameters()).device)

    num_tokens = 0

    while True:
        outputs = model(input_ids[:, -model.config.context_len:])
        next_token_logits = outputs[:, -1, :] / temperature
        if top_k is not None:
            # Keep only the k most likely tokens; the rest get zero probability.
            k = min(top_k, next_token_logits.shape[-1])
            kth_logit = torch.topk(next_token_logits, k, dim=-1).values[:, -1:]
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < kth_logit, float("-inf")
            )
        next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

        if next_token_id.item() == stop_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        num_tokens += 1

        if max_length > 0 and num_tokens >= max_length:
            break

    generated_text = tokenizer.decode(input_ids.squeeze().tolist())
    return generated_text
