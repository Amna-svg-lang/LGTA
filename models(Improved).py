import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Input, Bidirectional, Concatenate, LSTM, Dense,
    MultiHeadAttention, LayerNormalization, Dropout,
    RepeatVector, BatchNormalization, Flatten, Reshape
)
from keras.regularizers import l2
from keras.models import Model
from .helper import Sampling

# ---------------- Positional Embedding ---------------- #
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            shape=(self.max_seq_len, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="pos_embedding",
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_embedding[:seq_len, :]
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        return inputs + pos_encoding

# ---------------- Transformer Block ---------------- #
class TransformerBlockWithPosEncoding(keras.layers.Layer):
    def __init__(self, num_heads, ff_dim, max_seq_len, rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.rate = rate

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.pos_embed_layer = PositionalEmbedding(self.max_seq_len, embed_dim)
        self.mhatt = MultiHeadAttention(num_heads=self.num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(self.ff_dim, activation="relu", kernel_initializer="glorot_uniform"),
            Dense(embed_dim, kernel_initializer="glorot_uniform")
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.pos_embed_layer(inputs)
        attn_output = self.mhatt(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        context_vector = tf.reduce_mean(out2, axis=1)
        return context_vector

# ---------------- CVAE Model ---------------- #
class CVAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model,
                 window_size: int, kl_weight: float = 1.0,
                 lcr_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.window_size = window_size
        self.kl_weight = kl_weight
        self.lcr_weight = lcr_weight  # Weight for latent consistency regularization

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.lcr_loss_tracker = keras.metrics.Mean(name="lcr_loss")

    def call(self, inputs, training=None):
        dynamic_features, inp_data = inputs
        z_mean, z_log_var, z = self.encoder([dynamic_features, inp_data])
        pred = self.decoder([z, dynamic_features])
        return pred

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.lcr_loss_tracker
        ]

    def train_step(self, data):
        dynamic_features, inp_data = data

        # Apply small noise for latent consistency
        noise = tf.random.normal(tf.shape(inp_data), mean=0.0, stddev=0.01)
        inp_aug = inp_data + noise

        with tf.GradientTape() as tape:
            # Encode original input
            z_mean, z_log_var, z = self.encoder([dynamic_features, inp_data])
            pred = self.decoder([z, dynamic_features])

            # Encode perturbed input for latent consistency
            z_mean_aug, z_log_var_aug, z_aug = self.encoder([dynamic_features, inp_aug])

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(tf.square(inp_data - pred))

            # KL divergence
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Latent consistency loss (L2 distance between latent vectors)
            lcr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z - z_aug), axis=1))

            # Total loss
            total_loss = reconstruction_loss + self.kl_weight * kl_loss + self.lcr_weight * lcr_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lcr_loss_tracker.update_state(lcr_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "lcr_loss": self.lcr_loss_tracker.result(),
        }

# ---------------- Encoder-Decoder Factory ---------------- #
def get_CVAE(window_size: int, n_main_features: int, n_dyn_features: int, latent_dim: int,
             num_heads: int = 8, ff_dim: int = 256):
    # Encoder
    inp = Input(shape=(window_size, n_main_features), name="input_main_CVAE")
    dyn_inp = Input(shape=(window_size, n_dyn_features), name="input_dyn_CVAE")
    enc = Concatenate()([dyn_inp, inp])
    enc = Bidirectional(LSTM(n_main_features, kernel_initializer="glorot_uniform",
                             dropout=0.3, kernel_regularizer=l2(0.001),
                             return_sequences=True), merge_mode="ave")(enc)
    enc = BatchNormalization()(enc)
    enc = TransformerBlockWithPosEncoding(num_heads=num_heads, ff_dim=ff_dim,
                                          max_seq_len=window_size)(enc)
    enc = Dropout(0.3)(enc)
    enc = Dense(latent_dim, activation="relu", kernel_regularizer=l2(0.001))(enc)
    enc = BatchNormalization()(enc)
    z_mean = Dense(latent_dim)(enc)
    z_log_var = Dense(latent_dim)(enc)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model([dyn_inp, inp], [z_mean, z_log_var, z])

    # Decoder
    inp_z = Input(shape=(latent_dim,))
    dec_dyn_inp = Input(shape=(window_size, n_dyn_features))
    dec = RepeatVector(window_size)(inp_z)
    dec = Concatenate()([dec, dec_dyn_inp])
    dec = Bidirectional(LSTM(n_main_features, kernel_initializer="glorot_uniform",
                             return_sequences=True, dropout=0.3, kernel_regularizer=l2(0.001)), merge_mode="ave")(dec)
    out = Flatten()(dec)
    out = Dense(window_size * n_main_features)(out)
    out = Reshape((window_size, n_main_features))(out)
    decoder = Model([inp_z, dec_dyn_inp], out)

    return encoder, decoder
