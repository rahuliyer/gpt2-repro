import torch
import tiktoken

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = torch.tensor(
        tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
    ).unsqueeze(0).to(next(model.parameters()).device)

    num_tokens = 0

    while True:
        outputs = model(input_ids[:, -model.config.context_len:])
        next_token_logits = outputs[:, -1, :]
        next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

        if next_token_id.item() == tokenizer.eot_token:
            break

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        num_tokens += 1

        if max_length > 0 and num_tokens >= max_length:
            break

    generated_text = tokenizer.decode(input_ids.squeeze().tolist())
    return generated_text