"""Stop on generated text without decoding the growing output every step."""
from transformers import PreTrainedTokenizerBase, StopStringCriteria, StoppingCriteria


class GeneratedTextStop(StoppingCriteria):
    """Match only generated tokens, including markers split/merged across tokens.

    Transformers precomputes vocabulary overlaps and matches a bounded suffix
    with tensor operations. Minimal non-HF tokenizers used by replay/tests retain
    the original decode-based behavior, since they expose no vocabulary decoder.
    """

    def __init__(self, tokenizer, stop_strings, start_length=0):
        self.tokenizer = tokenizer
        self.stop_strings = tuple(stop_strings)
        self.start_length = start_length
        self.matcher = (StopStringCriteria(tokenizer, self.stop_strings)
                        if isinstance(tokenizer, PreTrainedTokenizerBase) else None)

    @property
    def suffix_length(self):
        return self.matcher.maximum_token_len if self.matcher is not None else None

    def __call__(self, input_ids, scores=None, **kwargs):
        generated = input_ids[:, self.start_length:]
        if self.matcher is not None:
            return self.matcher(generated, scores, **kwargs)
        text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
        return any(stop in text for stop in self.stop_strings)
