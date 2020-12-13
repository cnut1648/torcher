
    # return [line.split() for line in raw_text.split("\n")]
    # return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines if line]

#
#     corpus = []
#
# lines = read_time_machine()
#     tokens = tokenize(lines, 'char')
#     vocab = Vocab(tokens)
#     # Since each text line in the time machine dataset is not necessarily a
#     # sentence or a paragraph, flatten all the text lines into a single list
#     corpus = [vocab[token] for line in tokens for token in line]
#     if max_tokens > 0:
#         corpus = corpus[:max_tokens]
#     return corpus, vocab
