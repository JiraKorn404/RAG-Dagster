from functools import lru_cache

from transformers import AutoTokenizer


class Tokens:
    """Counts tokens with the embedding model's own tokenizer, so `max_tokens` means what it embeds."""

    def __init__(self, hf_tokenizer):
        self.hf = hf_tokenizer

    def count(self, text: str) -> int:
        return len(self.hf.encode(text, add_special_tokens=False))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """(start, end) character span of every token."""
        enc = self.hf(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]]


@lru_cache(maxsize=4)
def load_tokenizer(name: str) -> Tokens:
    return Tokens(AutoTokenizer.from_pretrained(name))
