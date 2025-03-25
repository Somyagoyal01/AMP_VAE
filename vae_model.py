import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import register_keras_serializable

# Define the Sampling layer (Reparameterization trick)
@register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the VAE as a Model subclass
@register_keras_serializable()
class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, amino_acids_count, max_sequence_length, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.amino_acids_count = amino_acids_count
        self.max_sequence_length = max_sequence_length

        # Encoder Layers
        self.encoder_embedding = layers.Embedding(input_dim=self.amino_acids_count + 1, output_dim=64)
        self.encoder_lstm = layers.LSTM(64, return_sequences=False)
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling_layer = Sampling()  # Instance of the Sampling layer

        # Decoder Layers
        self.decoder_dense1 = layers.Dense(64, activation='relu')
        self.decoder_repeat = layers.RepeatVector(int(self.max_sequence_length))
        self.decoder_lstm = layers.LSTM(64, return_sequences=True)
        self.decoder_dense2 = layers.TimeDistributed(layers.Dense(self.amino_acids_count + 1, activation='softmax'))

    def encode(self, x):
        embedding = self.encoder_embedding(x)
        lstm_out = self.encoder_lstm(embedding)
        z_mean = self.z_mean(lstm_out)
        z_log_var = self.z_log_var(lstm_out)
        z = self.sampling_layer([z_mean, z_log_var])  # Use the instance here
        return z_mean, z_log_var, z

    def decode(self, z):
        dense1 = self.decoder_dense1(z)
        repeat_vector = self.decoder_repeat(dense1)
        lstm_out = self.decoder_lstm(repeat_vector)
        reconstructed = self.decoder_dense2(lstm_out)
        return reconstructed

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decode(z)
        # Add KL divergence loss.
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss) / tf.cast(tf.shape(inputs)[1], dtype="float32"))  # Normalize KL loss by sequence length
        return reconstructed

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim,
            "amino_acids_count": self.amino_acids_count,
            "max_sequence_length": self.max_sequence_length
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)