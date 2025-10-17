#!/usr/bin/env python3
"""Transformer Application"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and prepares a dataset for machine translation"""

    def __init__(self):
        """Class constructor"""
        # Load Portuguese–English dataset
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # Tokenize both train and validation sets
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset"""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation pair into tokens with start/end tokens"""
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for encode() using tf.py_function"""
        result_pt, result_en = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        # Set shape (important for TensorFlow graph tracing)
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
