from torcherist.dataloader.text import *
from torcherist.dataset.text import read_ptb
import pytest
import numpy as np
import random

def test_sequential_text_sampler():
    my_seq = np.arange(35)
    for X, y in sequential_text_sampler(my_seq, batch_size=2, m=5):
        X, y = X.numpy(), y.numpy()
        assert np.all(X + 1 == y)
    prev_seq_end = None
    for X, y in sequential_text_sampler(my_seq, by="sequential", batch_size=2, m = 5):
        X, y = X.numpy(), y.numpy()
        if prev_seq_end is not None:
            assert np.all(prev_seq_end + 1 == X[:,0])
        prev_seq_end = X[:,-1]
        assert np.all(X + 1 == y)

def test_time_machine():
    data_iter, vocab = load_time_machine(batch_size=32,
                                         m=35,
                                         by="random")
    # 26 en char & empty space & <unk>
    assert len(vocab) == 28
    for X, y in data_iter:
        # for each batch
        for XX, yy in zip(X, y):
            XX, yy = XX.numpy().flatten(), yy.numpy().flatten()
            assert len(XX) == len(yy)
            assert np.all(XX[1:] == yy[:-1])

    data_iter, vocab = load_time_machine(batch_size=32,
                                         m=35,
                                         by="sequential")
    # 26 en char & empty space & <unk>
    assert len(vocab) == 28
    for X, y in data_iter:
        # for each batch
        for XX, yy in zip(X, y):
            XX, yy = XX.numpy().flatten(), yy.numpy().flatten()
            assert len(XX) == len(yy)
            assert np.all(XX[1:] == yy[:-1])

    # test if data_iter can be iter multiple times
    for X, y in data_iter:
        # for each batch
        for XX, yy in zip(X, y):
            XX, yy = XX.numpy().flatten(), yy.numpy().flatten()
            assert len(XX) == len(yy)
            assert np.all(XX[1:] == yy[:-1])
