import tiktoken

USER_START_TOKEN = "<|user-start|>"
USER_END_TOKEN = "<|user-end|>"
ASSISTANT_START_TOKEN = "<|assistant-start|>"
ASSISTANT_END_TOKEN = "<|assistant-end|>"
SYSTEM_START_TOKEN = "<|system-start|>"
SYSTEM_END_TOKEN = "<|system-end|>"

IGNORE_TOKEN_ID = -100

def get_tokenizer():
    base_tokenizer = tiktoken.get_encoding("gpt2", )

    special_tokens = {
        **base_tokenizer._special_tokens,
        USER_START_TOKEN: base_tokenizer.n_vocab,
        USER_END_TOKEN: base_tokenizer.n_vocab + 1,
        ASSISTANT_START_TOKEN: base_tokenizer.n_vocab + 2,
        ASSISTANT_END_TOKEN: base_tokenizer.n_vocab + 3,
        SYSTEM_START_TOKEN: base_tokenizer.n_vocab + 4,
        SYSTEM_END_TOKEN: base_tokenizer.n_vocab + 5,
    }

    tokenizer = tiktoken.Encoding(
        name="gpt2-chat",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens=special_tokens
    )

    return tokenizer