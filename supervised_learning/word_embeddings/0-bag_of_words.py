#!/usr/bin/env python3
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    Args:
        sentences (list of str): list of sentences to analyze
        vocab (list of str, optional): list of words to use as vocabulary
    Returns:
        embeddings (np.ndarray): bag-of-words matrix of shape (s, f)
        features (list of str): list of words used as features
    """
    processed_sentences = []
    for sentence in sentences:
        # Lowercase
        s = sentence.lower()
        # Replace possessive "'s" with nothing
        s = re.sub(r"'s\b", "", s)
        # Remove other punctuation
        s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
        words = s.split()
        processed_sentences.append(words)

    if vocab is None:
        vocab_set = set()
        for words in processed_sentences:
            vocab_set.update(words)
        vocab = sorted(vocab_set)

    features = np.array(vocab)
    word2idx = {word: i for i, word in enumerate(vocab)}

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, words in enumerate(processed_sentences):
        for word in words:
            if word in word2idx:
                embeddings[i, word2idx[word]] += 1

    return embeddings, features
