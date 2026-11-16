"""Per-question prompt tokens and Transformers cache for Plain decoding."""
from ..data.openseeker import pieces


class PlainGenerationState:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompt_ids = []
        self.encoded_messages = 0
        self.assistant_prompt = []
        self.past_key_values = None
        self.cached_ids = []

    def _encode(self, parts):
        return [token for text, _, _ in parts
                for token in self.tokenizer.encode(text, add_special_tokens=False)]

    def encode_turn(self, history):
        # The agent only appends complete messages. The assistant prefix is
        # temporary until the generated turn is serialized as a message.
        if not self.encoded_messages:
            self.assistant_prompt = self._encode(pieces((), generation_prompt=True))
        self.prompt_ids.extend(self._encode(pieces(history, start=self.encoded_messages)))
        self.encoded_messages = len(history)
        return {"input_ids": self.prompt_ids + self.assistant_prompt}

    def take_cache(self, input_ids):
        cache, prefix = self.past_key_values, self.cached_ids
        self.past_key_values, self.cached_ids = None, []
        # Detokenizing and serializing an assistant turn can change its token
        # boundaries. A hybrid cache cannot be rolled back to a shorter prefix.
        if prefix and len(prefix) < len(input_ids) and input_ids[:len(prefix)] == prefix:
            return cache, len(prefix)
        return None, 0

    def save_cache(self, cache, input_ids, generated):
        get_length = getattr(cache, "get_seq_length", None)
        if not callable(get_length):
            return
        length = int(get_length())
        if 0 < length <= len(input_ids) + len(generated):
            self.past_key_values = cache
            # Generation normally leaves the last sampled token unprocessed.
            self.cached_ids = (input_ids + generated)[:length]
