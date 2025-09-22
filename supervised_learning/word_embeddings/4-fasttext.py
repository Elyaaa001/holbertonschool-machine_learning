#!/usr/bin/env python3
"""
A function that creates and trains a gensim FastText model
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim FastText model.

    Parameters:
        sentences (list[list[str]]): Tokenized sentences.
        vector_size (int): Dimensionality of word vectors.
        min_count (int): Ignores words with total frequency lower than this.
        negative (int): Negative sampling size.
        window (int): Context window size.
        cbow (bool): If True, use CBOW; if False, use Skip-gram.
        iterations (int): Number of training epochs.
        seed (int): Random seed.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.FastText: The trained FastText model.
    """
    sg = 0 if cbow else 1
    model = gensim.models.FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
