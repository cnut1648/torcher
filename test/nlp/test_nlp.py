from torcherist.nlp import *
from torcherist.dataset.text import read_ptb
import pytest
from itertools import tee

doc = read_ptb()
vocab = Vocab(doc)


def test_count_text():
    counter = count_text(doc)
    assert counter["the"] == 50770


def test_Vocab():
    assert len(vocab) == 6719
    with pytest.raises(KeyError):
        vocab.token2idx["aer"]

    # aer rare word, map to unk
    assert vocab[["<unk>", "the", "aer"]] == [0, 1, 0]
    assert vocab.to_token([[0, 1], [0]]) == [["<unk>", "the"], ["<unk>"]]
    assert vocab.get_counts("the") == 50770
    assert vocab.get_counts("aer") == 1
    for sent in vocab.lookup(doc):
        assert "aer" not in sent
    corpus_gen = (token for sent in vocab.doc2corpus(doc) for token in sent)
    g1, g2 = tee(corpus_gen, 2)
    assert min(g1) == 0
    assert max(g2) == len(vocab) - 1
