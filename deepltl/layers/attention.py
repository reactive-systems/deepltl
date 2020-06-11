"""Implementation of scaled dot-product attention and multi-head attention as described in 'Attention Is All You Need' (Vaswani et al., 2017) based on https://www.tensorflow.org/tutorials/text/transformer"""


import tensorflow as tf


def scaled_dot_product_attention(queries, keys, values, mask=None, dtype=tf.float32):
    """
    Args:
        queries: (..., num_queries, d_queries)
        keys: (..., num_keys, d_queries)
        values: (..., num_keys, d_values)
        mask: (..., num_queries, num_keys)
    Returns:
        attention: (..., num_queries, d_values)
        attention_weights: (..., num_queries, num_keys)
    """
    attention_logits = tf.matmul(queries, keys, transpose_b=True)  # (..., num_queries, num_keys)

    # scale by square root of d_queries
    d_queries = tf.cast(tf.shape(queries)[-1], dtype)
    scaled_attention_logits = attention_logits / tf.math.sqrt(d_queries)

    # mask scaled values
    if mask is not None:
        scaled_attention_logits += (mask * dtype.min)

    # perform softmax over key axis and multiply resulting attention weights with values
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., num_queries, num_keys)
    attention = tf.matmul(attention_weights, values)  # (..., num_queries, d_values)
    return attention, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_embedding, num_heads):

        if d_embedding % num_heads != 0:
            raise ValueError(f"Embedding dimension {d_embedding} must be devisible by number of heads {num_heads}.")

        super().__init__()

        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_heads = d_embedding // num_heads

    def build(self, input_shape):

        self.Q = tf.keras.layers.Dense(self.d_embedding)
        self.K = tf.keras.layers.Dense(self.d_embedding)
        self.V = tf.keras.layers.Dense(self.d_embedding)

        self.final_projection = tf.keras.layers.Dense(self.d_embedding)

        super().build(input_shape)

    def split_heads(self, input, batch_size):
        """
        Splits last dimension d_embedding into (num_heads, d_heads) and transposes result
        Args:
            input: (batch_size, num_inputs, d_embedding)
        Returns:
            (batch_size, num_heads, num_inputs, d_heads)
        """
        input = tf.reshape(input, (batch_size, -1, self.num_heads, self.d_heads))
        return tf.transpose(input, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values, mask=None, cache=None):
        """
        Args:
            queries: (batch_size, num_queries, d_embedding)
            keys: (batch_size, num_keys, d_embedding)
            values: (batch_size, num_keys, d_embedding)
            mask: (batch_size, num_queries, num_keys)
            cache: a dictionary with attention from previous decoding steps that is used for fast decoding and has the following form:
                {'keys': [batch_size, i, num_heads, d_heads]
                 'values': [batch_size, i, num_heads, d_heads]}
                where i is the number of previous decoding steps
        Returns:
            attention: (batch_size, num_queries, d_embedding)
            attention_weights: (batch_size, num_queries, num_keys)
        """
        batch_size = tf.shape(queries)[0]

        queries = self.Q(queries)
        keys = self.K(keys)
        values = self.V(values)

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, num_queries, d_heads)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, num_keys, d_heads)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, num_keys, d_heads)

        if cache is not None:
            # concatenate cached keys and values
            keys = tf.concat([tf.cast(tf.transpose(cache['keys'], perm=[0, 2, 1, 3]), keys.dtype), keys], axis=2)
            values = tf.concat([tf.cast(tf.transpose(cache['values'], perm=[0, 2, 1, 3]), values.dtype), values], axis=2)
            # update cache
            cache['keys'] = tf.transpose(keys, perm=[0, 2, 1, 3])
            cache['values'] = tf.transpose(values, perm=[0, 2, 1, 3])

        scaled_attention, attention_weights = scaled_dot_product_attention(
            queries, keys, values, mask)  # (batch_size, num_heads, num_queries, d_heads) (batch_size, num_heads, num_queries, num_keys)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, num_queries, num_heads, d_heads)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_embedding))  # (batch_size, num_queries, d_embedding)
        attention = self.final_projection(concat_attention)  # (batch_size, num_queries, d_embedding)
        return attention, attention_weights


class SimpleMultNormAttention(tf.keras.layers.Layer):
    def __init__(self, units_size):
        super(SimpleMultNormAttention, self).__init__()
        self.units_size = units_size
        self.W = tf.keras.layers.Dense(units_size)

    def call(self, query, values):
        # query: (batch_size, units_size) = current dec state
        # values: (batch_size, seq_len, units_size) = enc outputs
        query_expanded = tf.expand_dims(query, 1) # (batch_size, 1, units_size)

        y_scaled = self.W(values)
        score = query_expanded * y_scaled
        score = tf.reduce_sum(score, axis=2, keepdims=True) # (batch_size, seq_len)
        score /= tf.sqrt(tf.constant(self.units_size, dtype=tf.float32))
        weights = tf.nn.softmax(score, axis=1)

        context_vector = weights * values # (batch_size, seq_len, units_size)
        context_vector = tf.reduce_sum(context_vector, axis=1) # (batch_size, units_size)
        return context_vector, weights
