from transformers import AutoTokenizer

def get_tokenizer(model_name="bert-base-uncased"):
    """Returns a HuggingFace tokenizer (free alternative to OpenAI tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
