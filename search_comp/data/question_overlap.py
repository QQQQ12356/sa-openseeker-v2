"""Match benchmark questions against training prompts, including embedded rewrites."""
from collections import Counter, defaultdict
import re


DEFAULT_THRESHOLD = 0.30


def words(text):
    return tuple(re.findall(r"\w+", text.casefold()))


def shingles(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))


class QuestionIndex:
    def __init__(self, questions, threshold=DEFAULT_THRESHOLD):
        if not 0 < threshold <= 1:
            raise ValueError("overlap threshold must be in (0, 1]")
        self.threshold = threshold
        self.questions = list(questions)
        self.exact = defaultdict(list)
        self.postings = defaultdict(list)
        self.tokens = []
        self.sizes = []
        for index, row in enumerate(self.questions):
            tokens = words(row["query"])
            self.tokens.append(tokens)
            self.exact[tokens].append(index)
            grams = shingles(tokens)
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings[gram].append(index)

    def matches(self, text):
        tokens = words(text)
        exact = set(self.exact.get(tokens, ()))
        grams = shingles(tokens)
        counts = Counter(index for gram in grams for index in self.postings.get(gram, ()))
        matches = [(index, 1.0, "exact") for index in exact]
        for index, shared in counts.items():
            if index in exact:
                continue
            target = self.tokens[index]
            if shared == self.sizes[index] and any(
                tokens[i:i + len(target)] == target for i in range(len(tokens) - len(target) + 1)
            ):
                matches.append((index, 1.0, "embedded"))
            elif shared >= 5 and shared / self.sizes[index] >= self.threshold:
                matches.append((index, shared / self.sizes[index], "four_gram"))
        return [(self.questions[i], score, method)
                for i, score, method in sorted(matches, key=lambda item: (-item[1], item[0]))]
