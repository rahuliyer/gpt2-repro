from .tokenizer import (
    USER_START_TOKEN,
    USER_END_TOKEN,
    ASSISTANT_START_TOKEN,
    ASSISTANT_END_TOKEN,
    SYSTEM_START_TOKEN,
    SYSTEM_END_TOKEN,
    IGNORE_TOKEN_ID
)

ROLE_TOKENS = {
    "system": (SYSTEM_START_TOKEN, SYSTEM_END_TOKEN),
    "user": (USER_START_TOKEN, USER_END_TOKEN),
    "assistant": (ASSISTANT_START_TOKEN, ASSISTANT_END_TOKEN),
}


def encode_message(tokenizer, message):
    """Encode one chat message, returning its input ids and labels.

    Only assistant messages are trained on: their content and end token are
    the labels, everything else is IGNORE_TOKEN_ID.
    """
    start_token, end_token = ROLE_TOKENS[message["role"]]
    start_id = tokenizer.encode_single_token(start_token)
    end_id = tokenizer.encode_single_token(end_token)
    encoded_content = tokenizer.encode(message["content"], disallowed_special=())

    inputs = [start_id] + encoded_content + [end_id]
    if message["role"] == "assistant":
        labels = [IGNORE_TOKEN_ID] + encoded_content + [end_id]
    else:
        labels = [IGNORE_TOKEN_ID] * len(inputs)

    return inputs, labels


def encode_prompt(tokenizer, messages):
    """Encode a conversation so the model's next tokens are the assistant reply."""
    inputs = []
    for message in messages:
        inputs += encode_message(tokenizer, message)[0]

    return inputs + [tokenizer.encode_single_token(ASSISTANT_START_TOKEN)]
