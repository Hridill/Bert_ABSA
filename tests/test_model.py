import pytest
import torch
from src.model import BERTBaseUncased
from src.config import MAX_LEN, TOKENIZER

def test_model_initialization():
    model = BERTBaseUncased()
    assert isinstance(model, torch.nn.Module)

def test_model_output_shape():
    model = BERTBaseUncased()
    # Create dummy input
    ids = torch.randint(0, 1000, (1, MAX_LEN))
    mask = torch.ones((1, MAX_LEN))
    token_type_ids = torch.zeros((1, MAX_LEN))
    
    output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
    assert output.shape == (1, 1)  # Batch size 1, output size 1

def test_tokenizer():
    test_text = "This is a test sentence"
    inputs = TOKENIZER.encode_plus(
        test_text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN
    )
    
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "token_type_ids" in inputs
    assert len(inputs["input_ids"]) <= MAX_LEN 