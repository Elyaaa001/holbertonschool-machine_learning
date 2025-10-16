#!/usr/bin/env python3
"""
Creates a Transformer network using Encoder and Decoder.
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer network combining encoder and decoder with final linear layer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Transformer constructor
        :param N: number of blocks in encoder and decoder
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

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        Forward pass for the Transformer.
        :param inputs: tensor (batch, input_seq_len)
        :param target: tensor (batch, target_seq_len)
        :param training: boolean, training mode
        :param encoder_mask: padding mask for encoder
        :param look_ahead_mask: look-ahead mask for decoder
        :param decoder_mask: padding mask for decoder
        :return: tensor (batch, target_seq_len, target_vocab)
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        output = self.linear(dec_output)
        return output
