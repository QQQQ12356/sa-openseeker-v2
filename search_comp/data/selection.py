"""Shared selection rules for measured trajectories (no tokenization or truncation)."""
from dataclasses import dataclass

from .openseeker import RESPONSE_RE


@dataclass
class Selection:
    min_length: int = 1
    max_length: int = 16384
    exact_length: int | None = None
    correctness: str = 'Correct'
    observations: str = 'any'
    tokenizer: str | None = None

    def __post_init__(self):
        if self.min_length < 1 or self.max_length < self.min_length:
            raise ValueError('length bounds must satisfy 1 <= min_length <= max_length')
        if self.exact_length is not None and not self.min_length <= self.exact_length <= self.max_length:
            raise ValueError('exact_length must lie inside the requested length bounds')
        if self.correctness not in ('Correct', 'Incorrect', 'any'):
            raise ValueError('invalid correctness filter')
        if self.observations not in ('any', 'with', 'without'):
            raise ValueError('invalid observations filter')

    def rejection(self, row):
        lengths = row['lengths']
        names = [self.tokenizer] if self.tokenizer else list(lengths)
        if not names or any(name not in lengths for name in names):
            raise ValueError('requested tokenizer has no length measurements')
        totals = [lengths[name]['total'] for name in names]
        if self.exact_length is not None and any(n != self.exact_length for n in totals):
            return 'not_exact_length'
        if any(n < self.min_length for n in totals):
            return 'under_length'
        if any(n > self.max_length for n in totals):
            return 'over_length'
        if self.correctness != 'any' and row.get('correctness') != self.correctness:
            return 'incorrect' if self.correctness == 'Correct' else 'correctness_mismatch'
        observed = any(match[1] for i, msg in enumerate(row['messages'])
                       if i > 1 and msg['role'] == 'user'
                       for match in RESPONSE_RE.finditer(msg['content']))
        if self.observations == 'with' and not observed:
            return 'no_observation'
        if self.observations == 'without' and observed:
            return 'has_observation'
        return None


def add_selection_arguments(parser):
    parser.add_argument('--min-length', type=int, default=1)
    parser.add_argument('--exact-length', type=int)
    parser.add_argument('--correctness', choices=['Correct', 'Incorrect', 'any'])
    parser.add_argument('--observations', choices=['any', 'with', 'without'], default='any')
    parser.add_argument('--tokenizer', help='Select length measurement; default requires all measured tokenizers to fit')


def selection_from_args(args):
    return Selection(args.min_length, args.max_length, args.exact_length,
                     args.correctness or ('Correct' if args.require_correct else 'any'),
                     args.observations, args.tokenizer)
