from transformers import GPT2LMHeadModel

from model import GPT2, GPT2LargeConfig, GPT2MediumConfig, GPT2SmallConfig

# Our config and the matching pretrained weights on the Hugging Face Hub.
MODEL_SIZES = {
    "small": (GPT2SmallConfig, "openai-community/gpt2"),
    "medium": (GPT2MediumConfig, "openai-community/gpt2-medium"),
    "large": (GPT2LargeConfig, "openai-community/gpt2-large"),
}

def build_model(model_size, vocab_size):
    config_class, hf_name = MODEL_SIZES[model_size]
    hf_model = GPT2LMHeadModel.from_pretrained(hf_name)
    model = GPT2.load_from_hf_model(config_class(), hf_model)

    model.extend_token_embeddings(vocab_size)

    return model
