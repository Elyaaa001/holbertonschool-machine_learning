#!/usr/bin/env python3
"""
Transformer model that combines an encoder and a decoder.
Provides a full Transformer layer with encoder, decoder, and final linear projection.
"""
import tensorflow as tf
import numpy as np
import random

# Set seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """
    Transformer model combining encoder, decoder, and final linear projection.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        :param N: number of encoder/decoder blocks
        :param dm: dimensionality of the model
        :param h: number of attention heads
        :param hidden: number of hidden units in feed-forward layers
        :param input_vocab: size of input vocabulary
        :param target_vocab: size of target vocabulary
        :param max_seq_input: maximum input sequence length
        :param max_seq_target: maximum target sequence length
        :param drop_rate: dropout rate
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Forward pass for the Transformer.
        :param inputs: tensor (batch, input_seq_len) for encoder
        :param target: tensor (batch, target_seq_len) for decoder
        :param training: boolean, training mode
        :param encoder_mask: encoder mask
        :param look_ahead_mask: first decoder attention mask
        :param decoder_mask: second decoder attention mask
        :return: tensor (batch, target_seq_len, target_vocab)
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        output = self.linear(dec_output)
        return output
