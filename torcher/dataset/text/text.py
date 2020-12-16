import io
import os
from typing import List
from zipfile import ZipFile

from nltk import word_tokenize, sent_tokenize

from torcher.config import dataset_dir

Sentence = List["word"]


def read_ptb(filepath=os.path.join(dataset_dir, "ptb.zip")) -> List[Sentence]:
    """penn treebank, note that they already have <unk> unknown token"""
    with ZipFile(filepath) as ptb_zip:
        with ptb_zip.open("ptb/ptb.train.txt") as f:
            raw_text = io.TextIOWrapper(f).read()
    return [line.split() for line in raw_text.split("\n")]


def read_time_machine(filepath=os.path.join(dataset_dir, "timemachine.txt")) -> List[Sentence]:
    with open(filepath, 'r') as f:
        raw_text = f.read()
    return [
        [token for token in word_tokenize(sent)]
        for sent in sent_tokenize(raw_text)
    ]
