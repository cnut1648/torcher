import io
import os
import re
from typing import List, Optional
from zipfile import ZipFile

from nltk import word_tokenize, sent_tokenize

from torcherist.config import dataset_dir

Sentence = List["word"]

dataset_dir = os.path.join(dataset_dir, "text")

def read_ptb(filepath=os.path.join(dataset_dir, "ptb.zip")) -> List[Sentence]:
    """penn treebank, note that they already have <unk> unknown token"""
    with ZipFile(filepath) as ptb_zip:
        with ptb_zip.open("ptb/ptb.train.txt") as f:
            raw_text = io.TextIOWrapper(f).read()
    return [line.split() for line in raw_text.split("\n")]


def read_time_machine(filepath=os.path.join(dataset_dir, "timemachine.txt")) -> List[Sentence]:
    """also see dataloader.text.load_time_machine"""
    with open(filepath, 'r') as f:
        raw_text = f.read()

    doc = []
    for sent in sent_tokenize(raw_text):
        # append empty space
        tokens = [
            (token, " ")
             for token in word_tokenize(
             re.sub('[^A-Za-z]+', ' ', sent).strip().lower()
            )
        ]
        doc.extend([token for tuple in tokens for token in tuple])
    return doc

def read_nmt(filepath=os.path.join(dataset_dir, "fra-eng.zip"),
             num_examples: Optional[int] = None):
    """machine translation: english-french"""
    with ZipFile(filepath) as ptb_zip:
        with ptb_zip.open("fra-eng/fra.txt") as f:
            raw_text = io.TextIOWrapper(f).read()

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = raw_text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    text = ''.join([
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ])

    # enize dataset
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target
