""" Transformer implementation based on https://github.com/tensorflow/models/tree/master/official/nlp/transformer"""

import tensorflow as tf

from deepltl.layers import attention
from deepltl.layers import positional_encoding as pe
from deepltl.models.beam_search import BeamSearch


def create_model(params, training, custom_pos_enc=False, attn_weights=False):
    """
    Args:
        params: dict, hyperparameter dictionary
        training: bool, whether model is called in training mode or not
        custom_pos_enc, bool, whether a custom postional encoding is provided as additional input
        attn_weights: bool, whether attention weights are part of the output
    """
    input = tf.keras.layers.Input((None,), dtype=tf.int32, name='input')
    transformer_inputs = {'input': input}
    model_inputs = [input]
    if custom_pos_enc:
        positional_encoding = tf.keras.layers.Input((None, None,), dtype=tf.float32, name='positional_encoding')
        transformer_inputs['positional_encoding'] = positional_encoding
        model_inputs.append(positional_encoding)
    if training:
        target = tf.keras.layers.Input((None,), dtype=tf.int32, name='target')
        transformer_inputs['target'] = target
        model_inputs.append(target)
    transformer = Transformer(params)
    if training:
        predictions, _ = transformer(transformer_inputs, training)
        predictions = TransformerMetricsLayer(params)([predictions, target])
        model = tf.keras.Model(model_inputs, predictions)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        mask = tf.cast(tf.math.logical_not(tf.math.equal(target, params['target_pad_id'])), params['dtype'])
        loss = tf.keras.layers.Lambda(lambda x: loss_object(x[0], x[1], x[2]))((target, predictions, mask))
        model.add_loss(loss)
        return model
    else:
        results = transformer(transformer_inputs, training)
        if attn_weights:
            outputs, scores, enc_attn_weights, dec_attn_weights = results['outputs'], results['scores'], results['enc_attn_weights'], results['dec_attn_weights']
            return tf.keras.Model(model_inputs, [outputs, scores, enc_attn_weights, dec_attn_weights])
        else:
            outputs, scores = results['outputs'], results['scores']
            return tf.keras.Model(model_inputs, [outputs, scores])


class TransformerMetricsLayer(tf.keras.layers.Layer):

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                dtype: tf.dtypes.Dtype(), datatype for floating point computations
                target_pad_id: int, encodes the padding token for targets
        """
        super(TransformerMetricsLayer, self).__init__()
        self.params = params

    def build(self, input_shape):
        self.accuracy_mean = tf.keras.metrics.Mean('accuracy')
        self.accuracy_per_sequence_mean = tf.keras.metrics.Mean('accuracy_per_sequence')
        super(TransformerMetricsLayer, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, inputs):
        predictions, targets = inputs[0], inputs[1]
        weights = tf.cast(tf.not_equal(targets, self.params['target_pad_id']), self.params['dtype'])
        outputs = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        targets = tf.cast(targets, tf.int32)

        # accuracy
        correct_predictions = tf.cast(tf.equal(outputs, targets), self.dtype)
        accuracy = self.accuracy_mean(*(correct_predictions, weights))
        self.add_metric(accuracy)

        # accuracy per sequence
        incorrect_predictions = tf.cast(tf.not_equal(outputs, targets), self.dtype) * weights
        correct_sequences = 1.0 - tf.minimum(1.0, tf.reduce_sum(incorrect_predictions, axis=-1))
        accuracy_per_sequence = self.accuracy_per_sequence_mean(correct_sequences, tf.constant(1.0))
        self.add_metric(accuracy_per_sequence)

        return predictions


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """A single encoder layer of the Transformer that consists of two sub-layers: a multi-head
    self-attention mechanism followed by a fully-connected feed-forward network. Both sub-layers
    employ a residual connection followed by a layer normalization."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerEncoderLayer, self).__init__()
        self.params = params

        self.multi_head_attn = attention.MultiHeadAttention(
            params['d_embed_enc'], params['num_heads'])

        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(params['d_ff'], activation=params['ff_activation']),
            tf.keras.layers.Dense(params['d_embed_enc'])
        ])

        self.norm_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_attn = tf.keras.layers.Dropout(params['dropout'])
        self.dropout_ff = tf.keras.layers.Dropout(params['dropout'])

    def call(self, input, mask, training):
        """
        Args:
            input: float tensor with shape (batch_size, input_length, d_embed_dec)
            mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: bool, whether layer is called in training mode or not
        """
        attn, attn_weights = self.multi_head_attn(input, input, input, mask)
        attn = self.dropout_attn(attn, training=training)
        norm_attn = self.norm_attn(attn + input)

        ff_out = self.ff(norm_attn)
        ff_out = self.dropout_ff(ff_out, training=training)
        norm_ff_out = self.norm_ff(ff_out + norm_attn)

        return norm_ff_out, attn_weights


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """A single decoder layer of the Transformer that consists of three sub-layers: a multi-head
    self-attention mechanism followed by a multi-head encoder-decoder-attention mechanism followed
    by a fully-connected feed-forward network. All three sub-layers employ a residual connection
    followed by a layer normalization."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerDecoderLayer, self).__init__()
        self.params = params

        self.multi_head_self_attn = attention.MultiHeadAttention(
            params['d_embed_dec'], params['num_heads'])
        self.multi_head_enc_dec_attn = attention.MultiHeadAttention(
            params['d_embed_dec'], params['num_heads'])

        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(params['d_ff'], activation=params['ff_activation']),
            tf.keras.layers.Dense(params['d_embed_dec'])
        ])

        self.norm_self_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_enc_dec_attn = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_self_attn = tf.keras.layers.Dropout(params['dropout'])
        self.dropout_enc_dec_attn = tf.keras.layers.Dropout(params['dropout'])
        self.dropout_ff = tf.keras.layers.Dropout(params['dropout'])

    def call(self, input, enc_output, look_ahead_mask, padding_mask, training, cache=None):
        """
        Args:
            input: float tensor with shape (batch_size, target_length, d_embed_dec)
            enc_output: float tensor with shape (batch_size, input_length, d_embed_enc)
            look_ahead_mask: float tensor with shape (1, 1, target_length, target_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: bool, whether layer is called in training mode or not
            cache: dict
        """
        self_attn, self_attn_weights = self.multi_head_self_attn(
            input, input, input, look_ahead_mask, cache)
        self_attn = self.dropout_self_attn(self_attn, training=training)
        norm_self_attn = self.norm_self_attn(self_attn + input)

        enc_dec_attn, enc_dec_attn_weights = self.multi_head_enc_dec_attn(norm_self_attn,
                                                                          enc_output, enc_output, padding_mask)
        enc_dec_attn = self.dropout_enc_dec_attn(
            enc_dec_attn, training=training)
        norm_enc_dec_attn = self.norm_enc_dec_attn(
            enc_dec_attn + norm_self_attn)

        ff_out = self.ff(norm_enc_dec_attn)
        ff_out = self.dropout_ff(ff_out, training=training)
        norm_ff_out = self.norm_ff(ff_out + norm_enc_dec_attn)

        return norm_ff_out, self_attn_weights, enc_dec_attn_weights


class TransformerEncoder(tf.keras.layers.Layer):
    """The encoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                input_vocab_size: int, size of input vocabulary
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerEncoder, self).__init__()
        self.params = params
        self.enc_layers = [TransformerEncoderLayer(params) for _ in range(params['num_layers'])]

    def call(self, x, padding_mask, training):
        attn_weights = {}
        for i in range(self.params['num_layers']):
            x, self_attn_weights = self.enc_layers[i](x, padding_mask, training)
            attn_weights[f'layer_{i+1}'] = {}
            attn_weights[f'layer_{i+1}']['self_attn'] = self_attn_weights
        return x, attn_weights


class TransformerDecoder(tf.keras.layers.Layer):
    """The decoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
                target_vocab_size: int, size of target vocabulary         
        """
        super(TransformerDecoder, self).__init__()
        self.params = params
        self.dec_layers = [TransformerDecoderLayer(params) for _ in range(params['num_layers'])]

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training, cache=None):
        attn_weights = {}
        for i in range(self.params['num_layers']):
            layer_cache = cache[f'layer_{i}'] if cache is not None else None
            x, self_attn_weights, enc_dec_attn_weights = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask, training, layer_cache)
            attn_weights[f'layer_{i+1}'] = {}
            attn_weights[f'layer_{i+1}']['self_attn'] = self_attn_weights
            attn_weights[f'layer_{i+1}']['enc_dec_attn'] = enc_dec_attn_weights
        return x, attn_weights


class Transformer(tf.keras.Model):
    """The Transformer that consists of an encoder and a decoder. The encoder maps the input
    sequence to a sequence of continuous representations. The decoder then generates an output
    sequence in an auto - regressive way."""

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                alpha: float, strength of normalization in beam search algorithm
                beam_size: int, number of beams kept by beam search algorithm
                d_embed_enc: int, dimension of encoder embedding
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layer
                input_vocab_size: int, size of input vocabulary
                max_encode_length: int, maximum length of input sequence
                max_decode_length: int, maximum lenght of target sequence
                dropout: float, percentage of droped out units
                dtype: tf.dtypes.Dtype(), datatype for floating point computations
                target_start_id: int, encodes the start token for targets
                target_eos_id: int, encodes the end of string token for targets
                target_vocab_size: int, size of target vocabulary
        """
        super(Transformer, self).__init__()
        self.params = params

        self.encoder_embedding = tf.keras.layers.Embedding(
            params['input_vocab_size'], params['d_embed_enc'])
        self.encoder_positional_encoding = pe.positional_encoding(
            params['max_encode_length'], params['d_embed_enc'])
        self.encoder_dropout = tf.keras.layers.Dropout(params['dropout'])

        self.encoder_stack = TransformerEncoder(params)

        self.decoder_embedding = tf.keras.layers.Embedding(
            params['target_vocab_size'], params['d_embed_dec'])
        self.decoder_positional_encoding = pe.positional_encoding(
            params['max_decode_length'], params['d_embed_dec'])
        self.decoder_dropout = tf.keras.layers.Dropout(params['dropout'])

        self.decoder_stack = TransformerDecoder(params)

        self.final_projection = tf.keras.layers.Dense(params['target_vocab_size'])
        self.softmax = tf.keras.layers.Softmax()

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, inputs, training):
        """
        Args:
            inputs: dictionary that contains the following (optional) keys:
                input: int tensor with shape (batch_size, input_length)
                (positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding)
                (target: int tensor with shape (batch_size, target_length))
            training: bool, whether model is called in training mode or not
        """
        input = inputs['input']

        input_padding_mask = create_padding_mask(input, self.params['input_pad_id'], self.params['dtype'])

        if 'positional_encoding' in inputs:
            positional_encoding = inputs['positional_encoding']
        else:
            seq_len = tf.shape(input)[1]
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]
        encoder_output, encoder_attn_weights = self.encode(input, input_padding_mask, positional_encoding, training)

        if 'target' in inputs:
            target = inputs['target']
            return self.decode(target, encoder_output, input_padding_mask, training)
        else:
            return self.predict(encoder_output, encoder_attn_weights, input_padding_mask, training)

    def encode(self, inputs, padding_mask, positional_encoding, training):
        """
        Args:
            inputs: int tensor with shape (batch_size, input_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc)
            training: boolean, specifies whether in training mode or not
        """
        input_embedding = self.encoder_embedding(inputs)
        input_embedding *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype']))
        input_embedding += positional_encoding
        input_embedding = self.encoder_dropout(input_embedding, training=training)
        encoder_output, attn_weights = self.encoder_stack(input_embedding, padding_mask, training)
        return encoder_output, attn_weights

    def decode(self, target, encoder_output, input_padding_mask, training):
        """
        Args:
            target: int tensor with shape (bath_size, target_length)
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        target_length = tf.shape(target)[1]
        look_ahead_mask = create_look_ahead_mask(target_length, self.params['dtype'])
        target_padding_mask = create_padding_mask(target, self.params['input_pad_id'], self.params['dtype'])
        look_ahead_mask = tf.maximum(look_ahead_mask, target_padding_mask)

        # shift targets to the right, insert start_id at first postion, and remove last element
        target = tf.pad(target, [[0, 0], [1, 0]], constant_values=self.params['target_start_id'])[:, :-1]

        target_embedding = self.decoder_embedding(target)  # (batch_size, target_length, d_embedding)
        target_embedding *= tf.math.sqrt(tf.cast(self.params['d_embed_dec'], self.params['dtype']))
        target_embedding += self.decoder_positional_encoding[:, :target_length, :]
        decoder_embedding = self.decoder_dropout(target_embedding, training=training)

        decoder_output, attn_weights = self.decoder_stack(
            decoder_embedding, encoder_output, look_ahead_mask, input_padding_mask, training)
        output = self.final_projection(decoder_output)
        probs = self.softmax(output)

        return probs, attn_weights

    def predict(self, encoder_output, encoder_attn_weights, input_padding_mask, training):
        """
        Args:
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            encoder_attn_weights: dictionary, self attention weights of the encoder
            input_padding_mask: flaot tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        batch_size = tf.shape(encoder_output)[0]

        def logits_fn(ids, i, cache):
            """
            Args:
                ids: int tensor with shape (batch_size * beam_size, index + 1)
                index: int, current index
                cache: dictionary storing encoder output, previous decoder attention values
            Returns:
                logits with shape (batch_size * beam_size, vocab_size) and updated cache
            """
            # set input to last generated id
            decoder_input = ids[:, -1:]
            decoder_input = self.decoder_embedding(decoder_input)
            decoder_input *= tf.math.sqrt(tf.cast(self.params['d_embed_dec'], self.params['dtype']))
            decoder_input += self.decoder_positional_encoding[:, i:i + 1, :]

            look_ahead_mask = create_look_ahead_mask(self.params['max_decode_length'], self.params['dtype'])
            self_attention_mask = look_ahead_mask[:, :, i:i + 1, :i + 1]
            decoder_output, attn_weights = self.decoder_stack(
                decoder_input, cache['encoder_output'], self_attention_mask, cache['input_padding_mask'], training, cache)

            output = self.final_projection(decoder_output)
            probs = self.softmax(output)
            probs = tf.squeeze(probs, axis=[1])
            return probs, cache

        initial_ids = tf.ones([batch_size], dtype=tf.int32) * self.params['target_start_id']

        num_heads = self.params['num_heads']
        d_heads = self.params['d_embed_dec'] // num_heads
        # create cache structure for decoder attention
        cache = {
            'layer_%d' % layer: {
                'keys': tf.zeros([batch_size, 0, num_heads, d_heads], dtype=self.params['dtype']),
                'values': tf.zeros([batch_size, 0, num_heads, d_heads], dtype=self.params['dtype'])
            } for layer in range(self.params['num_layers'])
        }
        # add encoder output to cache
        cache['encoder_output'] = encoder_output
        cache['input_padding_mask'] = input_padding_mask

        beam_search = BeamSearch(logits_fn, batch_size, self.params)
        decoded_ids, scores = beam_search.search(initial_ids, cache)

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        # compute attention weights
        _, decoder_attn_weights = self.decode(top_decoded_ids, encoder_output, input_padding_mask, training)

        return {'outputs': top_decoded_ids, 'scores': top_scores, 'enc_attn_weights': encoder_attn_weights, 'dec_attn_weights': decoder_attn_weights}


def create_padding_mask(input, pad_id, dtype=tf.float32):
    """
    Args:
        input: int tensor with shape (batch_size, input_length)
        pad_id: int, encodes the padding token
        dtype: tf.dtypes.Dtype(), data type of padding mask
    Returns:
        padding mask with shape (batch_size, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
    """
    mask = tf.cast(tf.math.equal(input, pad_id), dtype)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size, dtype=tf.float32):
    """
    creates a look ahead mask that masks future positions in a sequence, e.g., [[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]] for size 3
    Args:
        size: int, specifies the size of the look ahead mask
        dtype: tf.dtypes.Dtype(), data type of look ahead mask
    Returns:
        look ahead mask with shape (1, 1, size, size) that indicates masking with 1 and 0 everywhere else
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size), dtype), -1, 0)
    return tf.reshape(mask, [1, 1, size, size])
