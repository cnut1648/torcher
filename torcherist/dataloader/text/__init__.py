import random

from torcherist.nlp import Corpus, flatten_text, Vocab
from typing import List
import random
from torcherist.dataset.text import read_time_machine
import torch

def sequential_text_sampler(corpus: List[int],
                            by="random",
                            batch_size: int = 32,
                            m: int = 5):
    """
    sample sequential text by two strategy

    Args:
        corpus: list of index (should not be str), and should be flatten from Corpus
        by: strategy, either "random" or "sequential"
        m: integer, max len of each sample in one batch

    Returns:
        X: (batch_size, m), in total `num_subseqs` // `batch_size` number of batches
        y: (batch_size, m) y_ij is X_ij's next token

    """
    assert by in ["random", "sequential"], 'by must be either "random" or "sequential"'

    if by == "random":
        # Start with a random offset to partition a sequence
        # ignore first few tokens
        corpus = corpus[random.randint(0, m):]
        # Subtract 1 since we need to account for labels
        num_subseqs = (len(corpus) - 1) // m
        # The starting indices for subsequences of length `m`
        initial_indices = list(range(0, num_subseqs * m, m))
        # In random sampling, the subsequences from two adjacent random
        # minibatches during iteration are not necessarily adjacent on the
        # original sequence
        random.shuffle(initial_indices)

        def data(pos):
            # Return a sequence of length `m` starting from `pos`
            return corpus[pos: pos + m]

        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # Here, `initial_indices` contains randomized starting indices for
            # subsequences
            initial_indices_per_batch = initial_indices[i: i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            y = [data(j + 1) for j in initial_indices_per_batch]
            yield torch.tensor(X), torch.tensor(y)

    # sequential partition
    else:
        # start with a random offset (ignore first few tokens)
        offset = random.randint(0, m)
        # total number of tokens in all return X
        # Subtract 1 since we need to account for labels
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens]).reshape(batch_size, -1)
        ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens]).reshape(batch_size, -1)
        num_batches = Xs.shape[1] // m
        for i in range(0, num_batches * m, m):
            # ensure next batch is next to cur batch (sequential)
            yield Xs[:, i: i + m], ys[:, i: i + m]


def load_time_machine(batch_size: int, m: int,
                      by: str,
                      max_tokens = 10000,
                      *,
                      filepath: str = None):
    """
    load time machine text in character levels

    Args:
        batch_size: batch size
        m: generate batch of m-len characters
        by: sample strategy
        max_tokens: only load `max_tokens` chars from time machine
        filepath: path of time machine text

    Returns:
        data_iter: TimeMachineLoader, custom __iter__ so that can loop through data multiple time
        vocab
    """
    class TimeMachineLoader:
        def __init__(self, iter_fn: callable, *arg):
            self.iter_fn = iter_fn
            self. arg = arg
        def __iter__(self):
            return self.iter_fn(*self.arg)

    if filepath:
        doc = read_time_machine(filepath)
    else:
        doc = read_time_machine()

    # map to character level
    #  list of (list of char / word)
    doc = [list(token) for sent in doc for token in sent]
    vocab = Vocab(doc)

    corpus = list(flatten_text(vocab.doc2corpus(doc)))
    corpus = corpus[:max_tokens]

    return TimeMachineLoader(
        sequential_text_sampler,
        corpus,
        by,
        batch_size,
        m
    ), vocab
