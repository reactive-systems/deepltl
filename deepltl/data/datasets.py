
# pylint: disable = line-too-long

import os.path as path
import tensorflow as tf
from deepltl.data import ltl_parser
from deepltl.data.vocabulary import LTLVocabulary, TraceVocabulary


class LTLTracesDataset():
    """Dataset that consists of pairs of a LTL formula and a satisfying trace."""

    def __init__(self, name, ltl_vocab: LTLVocabulary, trace_vocab: TraceVocabulary, data_dir=None, format=None):
        """Given the name of the dataset tries to automatically determine data dir. Expects data file to have formula\ntrace\n format"""
        data_dir = data_dir if data_dir is not None else path.join(path.dirname(__file__), '../../../data')
        self.dataset_dir = path.join(data_dir, name)
        if not tf.io.gfile.exists(self.dataset_dir):
            raise FileNotFoundError('Cannot access dataset directory ' + str(self.dataset_dir))
        self.ltl_vocab = ltl_vocab
        self.trace_vocab = trace_vocab
        # format neeeds to be specified if tree positional encoding is used
        self.targets = ['train', 'val', 'test']

    def get_dataset(self, splits=None, dtype=tf.int32, max_length_formula=-1, max_length_trace=-1, prepend_start_token=True, tree_pos_enc=False):
        """Returns the requested spilts of the dataset or a dict containing the default ones."""
        if splits is not None:
            self.targets = splits
        res = {}
        for id, split in enumerate(self.targets):
            if tree_pos_enc:
                res[split] = tf.data.Dataset.from_generator(self._generator, (dtype, tf.float32, dtype), args=(id, max_length_formula, max_length_trace, prepend_start_token, tree_pos_enc))
            else:
                res[split] = tf.data.Dataset.from_generator(self._generator, (dtype, dtype), args=(id, max_length_formula, max_length_trace, prepend_start_token, tree_pos_enc))
        if splits is not None:
            res = [res[split] for split in splits]
        return res

    def _generator(self, split_id, max_length_formula, max_length_trace, prepend_start_token, tree_pos_enc):
        target_file = path.join(self.dataset_dir, self.targets[split_id] + '.txt')
        with tf.io.gfile.GFile(target_file, 'r') as file:  # expect formula\ntrace\n format
            for line_in in file:
                if line_in == '\n':
                    return
                line_out = next(file)  # get second line
                if max_length_formula >= 0 and len(line_in) > max_length_formula:
                    continue
                if max_length_trace >= 0 and len(line_out) > max_length_trace:
                    continue
                formula = ltl_parser.ltl_formula(line_in.strip(), 'network-polish')
                encoded_in = self.ltl_vocab.encode(formula.to_str('network-polish', spacing='all ops').split(' '))
                encoded_out = self.trace_vocab.encode(line_out.strip(), prepend_start_token=prepend_start_token)
                if tree_pos_enc:
                    position_list = formula.binary_position_list(format='lbt', add_first=True)
                    # pad to max length
                    max_length = max([len(l) for l in position_list])
                    padded_position_list = [l + [0] * (max_length - len(l)) for l in position_list]
                    yield (tf.constant(encoded_in), tf.constant(padded_position_list, dtype=tf.float32), tf.constant(encoded_out))
                else:
                    yield (tf.constant(encoded_in), tf.constant(encoded_out))


class BooleanSatDataset():
    def __init__(self, name, data_dir, formula_vocab=None, assignment_vocab=None):
        self.dataset_dir = path.join(data_dir, name)
        if not tf.io.gfile.exists(self.dataset_dir):
            raise FileNotFoundError('Cannot access dataset directory ' + str(self.dataset_dir))
        self.formula_vocab = formula_vocab
        self.assignment_vocab = assignment_vocab
        self.targets = ['train', 'val', 'test']
        self.feature_desc = {'formula_polish_tokens': tf.io.RaggedFeature(tf.int64), 'minimized_tokens': tf.io.RaggedFeature(tf.int64)}
        self.pos_encs = ['tree-branch-up', 'tree-branch-down']

    def get_dataset(self, splits=None, dtype=tf.int64, tree_pos_enc=False):
        if splits is not None:
            self.targets = splits
        res = {}
        for id, split in enumerate(self.targets):
            if tree_pos_enc:
                res[split] = tf.data.Dataset.from_generator(self._generator, (dtype, tf.float32, dtype), args=(id, tree_pos_enc))
            else:
                res[split] = tf.data.Dataset.from_generator(self._generator, (dtype, dtype), args=(id, tree_pos_enc))
        if splits is not None:
            res = [res[split] for split in splits]
        return res

    def _generator(self, split_id, tree_pos_enc):
        target_file = path.join(self.dataset_dir, self.targets[split_id] + '.txt')
        with tf.io.gfile.GFile(target_file, 'r') as file:  # expect formula\ntrace\n format
            for line_in in file:
                if line_in == '\n':
                    return
                line_out = next(file)  # get second line
                formula = ltl_parser.ltl_formula(line_in.strip(), 'network-polish')
                encoded_in = self.formula_vocab.encode(formula.to_str('network-polish', spacing='all ops').split(' '))
                encoded_out = self.assignment_vocab.encode(line_out)
                if tree_pos_enc:
                    position_list = formula.binary_position_list(format='lbt', add_first=True)
                    # pad to max length
                    max_length = max([len(l) for l in position_list])
                    padded_position_list = [l + [0] * (max_length - len(l)) for l in position_list]
                    yield (tf.constant(encoded_in), tf.constant(padded_position_list, dtype=tf.float32), tf.constant(encoded_out))
                else:
                    yield (tf.constant(encoded_in), tf.constant(encoded_out))


def teacher_forcing_map_fn(start_token, dtype=tf.int32):
    """ Takes (x, y) and maps it to ((x, <start>::y), y::0)"""
    def map_fn(x, y):
        y_s = tf.concat([tf.constant([start_token], dtype=dtype), y[:-1]], axis=-1)
        return ((x, y_s), y)
    return map_fn


def decoding_required_map_fn(start_token, dtype=tf.int32):
    """ Takes (x, y) and maps it to ((x, [<start>, 0, 0, ...]), y::0)"""
    def map_fn(x, y):
        y_s = tf.concat([tf.constant([start_token], dtype=dtype), tf.zeros_like(y[:-1])], axis=-1)
        return ((x, y_s), y)
    return map_fn
