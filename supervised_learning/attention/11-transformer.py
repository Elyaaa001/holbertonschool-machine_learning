#!/usr/bin/env python3
'''Transformer class using TensorFlow Keras'''

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder

class Transformer(tf.keras.Model):
    """Transformer network"""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Constructor
        Args:
            N: number of blocks in encoder and decoder
            dm: model dimensionality
            h: number of heads
            hidden: hidden units in feed-forward layers
            input_vocab: input vocabulary size
            target_vocab: target vocabulary size
            max_seq_input: max input sequence length
            max_seq_target: max target sequence length
            drop_rate: dropout rate
        Public attributes:
            encoder, decoder, linear
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden,
                               input_vocab, max_seq_input, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden,
                               target_vocab, max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Forward pass
        Args:
            inputs: tensor (batch, input_seq_len)
            target: tensor (batch, target_seq_len)
            training: boolean
            encoder_mask: padding mask for encoder
            look_ahead_mask: look-ahead mask for decoder
            decoder_mask: padding mask for decoder
        Returns:
            tensor (batch, target_seq_len, target_vocab)
        """
        # Encoder output
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Decoder output
        dec_output, _ = self.decoder(target, enc_output, training,
                                     look_ahead_mask, decoder_mask)

        # Final linear layer
        final_output = self.linear(dec_output)

        return final_output


if __name__ == "__main__":
    # Set seeds for reproducibility
    import os, random, numpy as np
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Instantiate transformer
    transformer = Transformer(6, 512, 8, 2048, 10000, 12000, 1000, 1500)

    # Print encoder, decoder, and final layer
    print(transformer.encoder)
    print(transformer.decoder)
    print(transformer.linear)

    # Fake input tensors
    x = tf.random.uniform((32, 10), dtype=tf.float32)
    y = tf.random.uniform((32, 15), dtype=tf.float32)

    # Forward pass
    output = transformer(x, y, training=True, encoder_mask=None,
                         look_ahead_mask=None, decoder_mask=None)

    # Print output shape and tensor
    print(output.shape)
    print(output)
