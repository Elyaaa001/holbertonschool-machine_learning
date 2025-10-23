#!/usr/bin/env python3
"""Machine Translation Dataset helper"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads the TED HRLR Portuguese→English dataset and prepares tokenizers.
    """

    def __init__(self):
        """
        - data_train: tf.data.Dataset train split (as_supervised=True)
        - data_valid: tf.data.Dataset validation split (as_supervised=True)
        - tokenizer_pt: pretrained Portuguese tokenizer
        - tokenizer_en: pretrained English tokenizer
        """
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True,
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates tokenizers using pretrained models.

        Args:
            data: tf.data.Dataset of (pt, en) pairs (kept for API compatibility)

        Returns:
            tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True
        )
        return tokenizer_pt, tokenizer_en
