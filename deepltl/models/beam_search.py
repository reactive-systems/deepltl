""" Beam Search implementation taken from https://github.com/tensorflow/models/blob/master/official/nlp/transformer/beam_search_v1.py"""

import tensorflow as tf
from tensorflow.python.util import nest


class StateKeys():
    """keys of the dictionary that stores the beam search state"""
    CUR_INDEX = 'cur_index'
    ALIVE_SEQ = 'alive_seq'
    ALIVE_LOG_PROBS = 'alive_log_probs'
    ALIVE_CACHE = 'alive_cache'
    FINISHED_SEQ = 'finished_seq'
    FINISHED_SCORES = 'finished_scores'
    FINISHED_FLAGS = 'finished_flags'


class BeamSearch():

    def __init__(self, logits_fn, batch_size, params):
        """
        Args:
            get_logits: interface to decoder
            batch_size: integer, batch size
            params: dictionary containing the following keys: alpha, beam_size, dtype, max_decode_length, target_eos_id, target_start_id, target_vocab_size
        """
        self.logits_fn = logits_fn
        self.batch_size = batch_size

        self.alpha = params['alpha']
        self.beam_size = params['beam_size']
        self.dtype = params['dtype']
        self.eos_id = params['target_eos_id']
        self.max_decode_length = params['max_decode_length']
        self.start_id = params['target_start_id']
        self.vocab_size = params['target_vocab_size']

    def search(self, initial_ids, initial_cache):
        """
        Args:
            initial cache: dictionary storing cached values to be passed into logits_fn
        Returns:
            top decoded sequences with shape (batch_size, beam_size, max_decode_length)
            scores of top sequences with shape (batch_size, beam_size)
        """
        # get initial state
        state, state_shapes = self.get_initial_state(initial_ids, initial_cache)

        finished_state = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(cond=self.condition, body=self.step, loop_vars=[state], shape_invariants=[state_shapes], parallel_iterations=1))
        finished_state = finished_state[0]

        alive_seq = finished_state[StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[StateKeys.FINISHED_FLAGS]

        finished_cond = tf.reduce_any(finished_flags, 1, name='finished_cond')
        seq_cond = expand_to_same_rank(finished_cond, finished_seq)
        score_cond = expand_to_same_rank(finished_cond, finished_scores)

        # if there are no finished sequences for a batch item return alive sequences
        finished_seq = tf.where(seq_cond, finished_seq, alive_seq)
        finished_scores = tf.where(score_cond, finished_scores, alive_log_probs)

        return finished_seq, finished_scores

    def get_initial_state(self, initial_ids, initial_cache):
        """
        Args:
            initial cache: dictionary storing cached values to be passed into the logits_fn
        Returns:
            initial state
            shape invariants
        """
        cur_index = tf.constant(0)

        alive_seq = self.expand_to_beam_size(initial_ids)
        alive_seq = tf.expand_dims(alive_seq, axis=2)

        initial_log_probs = tf.constant([[0.] + [self.dtype.min] * (self.beam_size - 1)], dtype=self.dtype, name='initial_log_probs')
        alive_log_probs = tf.tile(initial_log_probs, [self.batch_size, 1], name='alive_log_probs')

        alive_cache = nest.map_structure(lambda t: self.expand_to_beam_size(t), initial_cache)

        finished_seq = tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)
        finished_scores = tf.ones([self.batch_size, self.beam_size], dtype=self.dtype) * self.dtype.min
        finished_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)

        state = {
            StateKeys.CUR_INDEX: cur_index,
            StateKeys.ALIVE_SEQ: alive_seq,
            StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
            StateKeys.ALIVE_CACHE: alive_cache,
            StateKeys.FINISHED_SEQ: finished_seq,
            StateKeys.FINISHED_FLAGS: finished_flags,
            StateKeys.FINISHED_SCORES: finished_scores
        }

        state_shape = {
            StateKeys.CUR_INDEX: tf.TensorShape([]),
            StateKeys.ALIVE_SEQ: tf.TensorShape([None, self.beam_size, None]),
            StateKeys.ALIVE_LOG_PROBS: tf.TensorShape([None, self.beam_size]),
            StateKeys.ALIVE_CACHE: nest.map_structure(get_shape_keep_last_dim, alive_cache),
            StateKeys.FINISHED_SEQ: tf.TensorShape([None, self.beam_size, None]),
            StateKeys.FINISHED_FLAGS: tf.TensorShape([None, self.beam_size]),
            StateKeys.FINISHED_SCORES: tf.TensorShape([None, self.beam_size])
        }

        return state, state_shape

    def condition(self, state):
        """
        Args:
            state: current state
        Returns:
            bool tensor, whether beam search should be continued or not
        """
        # check whether maximum decode length has been reached
        cur_index = state[StateKeys.CUR_INDEX]
        not_at_max_decode_length = tf.less(cur_index, self.max_decode_length)

        # check whether worst score in finished sequences is better than the best score in alive sequences such that no improvements are possible

        alive_log_probs = state[StateKeys.ALIVE_LOG_PROBS]
        finished_scores = state[StateKeys.FINISHED_SCORES]
        finished_flags = state[StateKeys.FINISHED_FLAGS]

        # get best scores in alive sequences
        max_length_norm = self.length_normalization(self.alpha, self.max_decode_length)
        best_alive_scores = alive_log_probs[:, 0] / max_length_norm

        # get worst scores in finished sequences
        finished_scores *= tf.cast(finished_flags, self.dtype)  # set filler scores to zero
        worst_finished_scores = tf.reduce_min(finished_scores, axis=1)
        finished_batches = tf.reduce_any(finished_flags, axis=1)
        worst_finished_scores += ((1.0 - tf.cast(finished_batches, self.dtype)) * self.dtype.min)  # if there are no finished sequences set to large negative number

        worst_finished_better_than_best_alive = tf.reduce_all(tf.greater(worst_finished_scores, best_alive_scores))

        return tf.logical_and(not_at_max_decode_length, tf.logical_not(worst_finished_better_than_best_alive))

    def step(self, state):
        """
        Args:
            state: dictionary, current state
        Returns:
            new state
        """
        # grow alive sequences by one step each
        new_alive_seq, new_alive_log_probs, top_ids, new_alive_cache = self.grow_alive_seq(state)

        new_finished_flags = tf.equal(top_ids, self.eos_id)

        # get new alive and finished state
        alive_state = self.get_new_alive_state(new_alive_seq, new_alive_log_probs, new_finished_flags, new_alive_cache)
        finished_state = self.get_new_finished_state(state, new_alive_seq, new_alive_log_probs, new_finished_flags)

        # construct new state
        new_state = {StateKeys.CUR_INDEX: state[StateKeys.CUR_INDEX] + 1}
        new_state.update(alive_state)
        new_state.update(finished_state)
        return [new_state]

    def grow_alive_seq(self, state):
        """
        Args:
            state: dictionary, current state
        Returns:
            top sequences with shape (batch_size, 2 * beam_size, cur_index + 1)
            scores of top sequences with shape (batch_size, 2 * beam_size)
            new cache of the top sequences
        """

        cur_index = state[StateKeys.CUR_INDEX]
        alive_seq = state[StateKeys.ALIVE_SEQ]

        alive_log_probs = state[StateKeys.ALIVE_LOG_PROBS]
        alive_cache = state[StateKeys.ALIVE_CACHE]

        flat_ids = tf.reshape(alive_seq, [self.batch_size * self.beam_size, -1])
        flat_cache = nest.map_structure(self.flatten_beam_dim, alive_cache)

        flat_logits, flat_cache = self.logits_fn(flat_ids, cur_index, flat_cache)

        logits = tf.reshape(flat_logits, [self.batch_size, self.beam_size, -1])

        new_cache = nest.map_structure(lambda t: self.unflatten_beam_dim(t), flat_cache)

        # convert logits to normalized log probs
        candidate_log_probs = logits - tf.reduce_logsumexp(logits, axis=2, keepdims=True)

        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)  # (batch_size, beam_size, vocab_size)

        # get the 2 * beam_size candidates with the hinghest probabilities
        flat_log_probs = tf.reshape(log_probs, [-1, self.beam_size * self.vocab_size])  # (batch_size, beam_size * vocab_size)

        topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, 2 * self.beam_size)

        # extraxt alive sequences with highest log probabilities
        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = self.gather_beams([alive_seq, new_cache], topk_beam_indices, 2 * self.beam_size)
        topk_ids = topk_indices % self.vocab_size
        topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

        return topk_seq, topk_log_probs, topk_ids, new_cache

    def get_new_alive_state(self, new_alive_seq, new_alive_log_probs, new_finished_flags, new_alive_cache):
        """
        Args:
            new_alive_seq: int32 tensor, new grown sequences with shape (batch_size, 2 * beam_size, cur_index + 1)
            new_alive_log_probs: dtype tensor, log probabilities of new sequences with shape (batch_size, 2 * beam_size)
            new_finished_flags: bool tensor, indicates which sequences are alive
            new_alive_cache: dict, new cache of sequences
        Returns:
            new partial state containing:
                top sequences that are still alive with shape (batch_size, beam_size, cur_index + 1)
                log probabilities of top alive sequences with shape (batch_size, beam_size)
                cache of top alive sequences
        """
        # set finished sequences to large negative number
        new_alive_log_probs += tf.cast(new_finished_flags, self.dtype) * self.dtype.min

        top_alive_seq, top_alive_log_probs, top_alive_cache = self.gather_top_beams([new_alive_seq, new_alive_log_probs, new_alive_cache], new_alive_log_probs, self.beam_size)

        return {
            StateKeys.ALIVE_SEQ: top_alive_seq,
            StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
            StateKeys.ALIVE_CACHE: top_alive_cache
        }

    def get_new_finished_state(self, state, new_alive_seq, new_alive_log_probs, new_finished_flags):
        """
        Args:
            state: dictionary, current state
            new_alive_seq: int32 tensor, new grown sequences with shape (batch_size, 2 * beam_size, cur_index + 1)
            new_alive_log_probs: dtype tensor, log probabilities of new sequences with shape (batch_size, 2 * beam_size)
            new_finished_flags: bool tensor, indicates which sequences are alive
        Returns:
            new partial state containing:
                top finished sequences with shape (batch_size, beam_size, cur_index + 1)
                finished scores of top finished sequences with shape (batch_size, beam_size)
                finished flags of finished sequences with shape (batch_size, beam_size)
        """
        cur_index = state[StateKeys.CUR_INDEX]

        finished_seq = state[StateKeys.FINISHED_SEQ]
        finished_scores = state[StateKeys.FINISHED_SCORES]
        finished_flags = state[StateKeys.FINISHED_FLAGS]

        # append a column of zeros to finished_seq to increment length
        finished_seq = tf.concat([finished_seq, tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)], axis=2)

        # calculate new scores from log probabilities
        length_norm = self.length_normalization(self.alpha, cur_index + 1)
        new_scores = new_alive_log_probs / length_norm
        new_scores += (1. - tf.cast(new_finished_flags, self.dtype)) * self.dtype.min

        finished_seq = tf.concat([finished_seq, new_alive_seq], axis=1)
        finished_scores = tf.concat([finished_scores, new_scores], axis=1)
        finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

        top_finished_seq, top_finished_scores, top_finished_flags = self.gather_top_beams([finished_seq, finished_scores, finished_flags], finished_scores, self.beam_size)

        return {
            StateKeys.FINISHED_SEQ: top_finished_seq,
            StateKeys.FINISHED_SCORES: top_finished_scores,
            StateKeys.FINISHED_FLAGS: top_finished_flags
        }

    def gather_beams(self, nested, beam_indices, new_beam_size):
        """
        Args:
            nested: nested structure (tensor, list, tuple or dict) containing tensors with shape (batch_size, beam_size, ...)
            beam_indices: tensor with shape (batch_size, new_beam_size) specifying beams that are gathered
            new_beam_size: number of beams pulled from nested tensors
        Returns:
            nested structure containing tensors with shape (batch_size, new_beam_size, ...)
        """
        batch_pos = tf.range(self.batch_size * new_beam_size) // new_beam_size
        batch_pos = tf.reshape(batch_pos, [self.batch_size, new_beam_size])  # [[0,0,0,...],[1,1,1,...],...]

        # creating a tensor with shape (batch_size, beam_size, 2) where the last dimension constains gathering coordinates (i, j)
        coordinates = tf.stack([batch_pos, beam_indices], axis=2)

        return nest.map_structure(lambda state: tf.gather_nd(state, coordinates), nested)

    def gather_top_beams(self, nested, log_probs, beam_size):
        _, top_indices = tf.nn.top_k(log_probs, k=beam_size)
        return self.gather_beams(nested, top_indices, beam_size)

    def length_normalization(self, alpha, length):
        return tf.pow(tf.cast(length, self.dtype), alpha)

    def expand_to_beam_size(self, tensor):
        tensor = tf.expand_dims(tensor, axis=1)
        tile_dims = [1] * tensor.shape.ndims
        tile_dims[1] = self.beam_size
        return tf.tile(tensor, tile_dims)

    def flatten_beam_dim(self, tensor):
        shape = shape_list(tensor)
        shape[0] *= shape[1]
        shape.pop(1)
        return tf.reshape(tensor, shape)

    def unflatten_beam_dim(self, tensor):
        shape = shape_list(tensor)
        new_shape = [self.batch_size, self.beam_size] + shape[1:]
        return tf.reshape(tensor, new_shape)

    def get_shape(self, tensor):
        return tf.TensorShape(shape_list(tensor))


def shape_list(tensor):
    shape = tensor.get_shape().as_list()
    dynamic_shape = tf.shape(tensor)
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape


def get_shape_keep_last_dim(tensor):
    shape = shape_list(tensor)
    for i in range(len(shape) - 1):
        shape[i] = None
    if isinstance(shape[-1], tf.Tensor):
        shape[-1] = None
    return tf.TensorShape(shape)


def expand_to_same_rank(tensor, target):
    if tensor.shape.rank is None:
        raise ValueError('')
    if target.shape.rank is None:
        raise ValueError('')
    diff_rank = target.shape.rank - tensor.shape.rank
    for _ in range(diff_rank):
        tensor = tf.expand_dims(tensor, -1)
    return tensor
