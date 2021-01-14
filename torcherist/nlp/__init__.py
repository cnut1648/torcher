import random
from collections import Counter
from typing import List, Union, Iterator

import torch

Sentence = List["word"]
Doc = List[Sentence]

# sentences, but in index
Corpus = List[List[int]]

# Doc or Corpus, list of list of (idx or raw str)
Text = Union[Doc, Corpus]

def flatten_text(text: Text) -> Iterator[Union[int, str]]:
    """flatten text into one list"""
    for sent in text:
        for token in sent:
            yield token

def count_text(text: Text) -> Counter:
    """:return counter of each token in the text"""
    return Counter(
        flatten_text(text)
    )

class Vocab:
    def __init__(self, doc: Doc,
                 cutoff=10,
                 unk_label="<unk>",
                 reserved_tokens: List[str] = None):
        if reserved_tokens is None:
            reserved_tokens = []
        self.unk_label = unk_label

        # sort based on freq
        #   still reserve freq ≤ cutoff
        self.token_freq = count_text(doc)

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

    def lookup(self, doc: Doc) -> Doc:
        """map rare words to unk_label, if any"""
        # first map to idx (if rare or not exist map to 0)
        #      then get token
        return [[self.to_token(self[tk]) for tk in sent]
                for sent in doc]

    def doc2corpus(self, doc: Doc) -> Corpus:
        """map token (str) to corpus (idx)"""
        # first map rare then map to index
        return [self[sentence] for sentence in self.lookup(doc)]


def ngram(text: Text, n: int):
    text_flatten = list(flatten_text(text))
    return [text_flatten[i:i+n] for i in range(len(text_flatten)+1)]
