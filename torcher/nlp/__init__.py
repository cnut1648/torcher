import random
from collections import Counter
from typing import List

import torch

# def List[Sentence] as doc
Sentence = List["word"]

# sentences, but in index
Corpus = List[List[int]]


def count_corpus(doc: List[Sentence]):
    """:return counter of each token in the doc"""
    return Counter(
        token for sent in doc
        for token in sent
    )


class Vocab:
    def __init__(self, doc: List[Sentence],
                 cutoff=10,
                 unk_label="<unk>",
                 reserved_tokens: List[str] = None):
        if reserved_tokens is None:
            reserved_tokens = []
        self.unk_label = unk_label

        # sort based on freq
        #   still reserve freq ≤ cutoff
        self.token_freq = count_corpus(doc)

        # in case doc contains unk_label symbol, not using enumerate
        #    (since index would discontinuous)
        #    get the list and update idx2token with list
        # most freq token to lower idx number
        #   unk index = 0, then reserved tokens, then others
        uniq_tokens = [unk_label] + reserved_tokens
        uniq_tokens += [token
                        for (token, ct) in self.token_freq.most_common()
                        if ct >= cutoff and token not in uniq_tokens]
        self.idx2token = {
            i: token for i, token in enumerate(uniq_tokens)
        }
        self.token2idx = dict(map(reversed, self.idx2token.items()))

    def __len__(self):
        """only the len of unique tokens (not counting freq ≤ cutoff)"""
        return len(self.idx2token)

    def __getitem__(self, token):
        """return the index of token, token str or iterable of str
            if not in token2idx (either freq ≤ cutoff or not in vocab)
                mapped to unk_label (0)"""
        if isinstance(token, (list, tuple)):
            return list(map(self.__getitem__, token))
        return self.token2idx.get(token, 0)

    def to_token(self, index):
        """return the token from indice, index int or iterable of int"""
        if isinstance(index, (list, tuple)):
            return list(map(self.to_token, index))
        return self.idx2token[index]

    def get_counts(self, token):
        """return the freq of token (still preserve freq ≤ cutoff),
         token str or iterable of str"""
        if isinstance(token, (list, tuple)):
            return list(map(self.token_freq.get, token))
        return self.token_freq[token]

    def lookup(self, doc: List[Sentence]) -> List[Sentence]:
        """map rare words to unk_label, if any"""
        # first map to idx (if rare or not exist map to 0)
        #      then get token
        return [[self.to_token(self[tk]) for tk in sent]
                for sent in doc]

    def doc2corpus(self, doc: List[Sentence]) -> Corpus:
        """map token (str) to corpus (idx)"""
        # first map rare then map to index
        return [self[sentence] for sentence in self.lookup(doc)]


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset to partition a sequence
    corpus = corpus[random.randint(0, num_steps):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_subseqs_per_example = num_subseqs // batch_size
    for i in range(0, batch_size * num_subseqs_per_example, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
