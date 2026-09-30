import pytest
import torch
from transformers import GPT2LMHeadModel

from model import GPT2, GPT2SmallConfig


@pytest.mark.integration
def test_gpt2_matches_hugging_face():
    config = GPT2SmallConfig()

    hf_model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2"
    )
    hf_model.eval()

    model = GPT2.load_from_hf_model(config, hf_model)
    model.eval()

    generator = torch.Generator().manual_seed(42)
    inputs = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 32),
        generator=generator,
    )

    with torch.inference_mode():
        expected = hf_model(inputs).logits
        actual = model(inputs)

    assert actual.shape == expected.shape
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
