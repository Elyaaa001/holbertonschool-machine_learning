#!/usr/bin/env python3
"""
A fully deterministic function that creates and trains a gensim FastText model
"""
import os
import numpy as np
import gensim

# Ensure reproducibility for BLAS/MKL/OpenMP parallel threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
np.random.seed(1)


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=1, workers=1):
    """
    Creates and trains a gensim FastText model in a deterministic way.

    Parameters:
        sentences (list[list[str]]): Tokenized sentences.
        vector_size (int): Dimensionality of word vectors.
        min_count (int): Ignores words with total frequency lower than this.
        negative (int): Negative sampling size.
        window (int): Context window size.
        cbow (bool): If True, use CBOW; if False, use Skip-gram.
        iterations (int): Number of training epochs.
        seed (int): Random seed.
        workers (int): Number of worker threads (1 recommended for reproducibility).

    Returns:
        gensim.models.FastText: The trained FastText model.
    """
    sg = 0 if cbow else 1  # CBOW=0, Skip-gram=1

    # Initialize the FastText model
    model = gensim.models.FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )

    # Build vocabulary
    model.build_vocab(sentences)

    # Train the model
    model.train(sentences, total_examples=model.corpus_count, epochs=iterations)

    return model


# === Example usage ===
if __name__ == "__main__":
    from gensim.test.utils import common_texts

    # Train the model
    model = fasttext_model(common_texts, vector_size=50, iterations=10, seed=1, workers=1)

    # Check example output
    print(common_texts[:2])
    print(model.wv['computer'])
