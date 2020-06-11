# Content of this file is largely taken from files in the examples/random_formulas directory of py-aiger: https://github.com/MarkusRabe/py-aiger
# which is licensed under the following:
#
# MIT License
#
# Copyright (c) 2018 Marcell Vazquez-Chanlat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


# pylint: disable = line-too-long

import os, sys
import random
import argparse

import spot
import aiger
import aiger_sat

from deepltl.data import ltl_parser
from deepltl.data.generator import DistributionGate
from deepltl.utils import utils


## Generator things
def pre_order(tree):
    to_process = [tree]
    while to_process:
        elem = to_process.pop()
        if isinstance(elem, list):
            to_process.extend(elem[::-1])
        else:
            assert isinstance(elem, str) or isinstance(elem, bool)
            yield elem

def flatten_to_string(f):
    return ' '.join(map(str, pre_order(f)))

class Generator():
    def __init__(self, num_variables=10, bool_constants=['True', 'False'], unary_operators=['neg'], binary_operators=['or', 'and', 'xor', 'eq']):
        self.num_variables = num_variables
        self.variables = ['var%02d' % v for v in range(self.num_variables)]
        self.bool_constants = bool_constants.copy()
        self.unary_operators = unary_operators.copy()
        self.binary_operators = binary_operators.copy()

    def to_expression(self, token_sequence): # pre-ordered list
        # if isinstance(token_sequence, list):
        #   token_sequence = iter(pre_order(token_sequence))
        token_sequence = iter(token_sequence)
        elem = next(token_sequence, None)
        if elem is None:
            raise ValueError('Sequence ends before expression is complete.')
        if elem in self.variables:
            return aiger.atom(elem)
        if elem in self.bool_constants:
            return aiger.atom(elem == 'True')
        if elem in self.unary_operators:
            assert elem == 'neg'
            return ~ self.to_expression(token_sequence)
        assert elem in self.binary_operators, 'Unknown op: %s' % elem
        left = self.to_expression(token_sequence)
        right = self.to_expression(token_sequence)
        if elem == 'or':
            return left | right
        if elem == 'and':
            return left & right
        if elem == 'xor':
            return left ^ right
        if elem == 'eq':
            return left == right
        raise ValueError('Should not reach this point')


## Model things
def get_model(formula):
    solver = aiger_sat.SolverWrapper()
    solver.add_expr(formula)
    return solver.get_model()

def minimize_model(formula, model):
    if model is None:
        return None
    solver = aiger_sat.SolverWrapper()
    solver.add_expr(~formula)
    if solver.is_sat(assumptions=model):
        raise ValueError('UNSAT core generation failed.')
    minimized = solver.get_unsat_core()
    if minimized is None:
        minimized = {}
    return minimized

def generate_model(formula_pre, generator=None):
    if generator is None:
        generator = Generator()
    expr = generator.to_expression(formula_pre)
    solver = aiger_sat.SolverWrapper()
    try:
        solver.add_expr(expr)
    except ValueError as e:
        print('ValueError', str(e))
        return None
    model = solver.get_model()
    model_str = None
    if model:
        model_word_pairs = ['%s %s' % x for x in sorted(model.items(), key=lambda x: x[0])]
        model_str = ' '.join(model_word_pairs)

    minimized = minimize_model(expr, model)
    minimized_str = None
    if minimized:
        minimized_word_pairs = ['%s %s' % x for x in sorted(minimized.items(), key=lambda x: x[0])]
        minimized_str = ' '.join(minimized_word_pairs)
    return minimized_str


def is_model(polish_formula, model, generator=None):
    if generator is None:
        generator = Generator()
    formula = generator.to_expression(polish_formula)
    solver = aiger_sat.SolverWrapper()
    solver.add_expr(~formula)
    return not solver.is_sat(assumptions=model)


## pyaiger <-> spot conversion things
spot_to_pyaiger_dict = {'1':'True', '0':'False', '!':'neg', '<->':'eq', 'xor':'xor', '&':'and', '|':'or'}
pyaiger_to_spot_dict = {val : key for key, val in spot_to_pyaiger_dict.items()}

def spot_to_pyaiger(token_list):
    res = []
    for token in token_list:
        if token in spot_to_pyaiger_dict:
            res.append(spot_to_pyaiger_dict[token])
        else:
            n = ord(token) - 97
            if n >= 26:
                raise ValueError()
            res.append(f'var{n:02}')
    return res

def pyaiger_to_spot(token_list):
    res = []
    for token in token_list:
        if token in pyaiger_to_spot_dict:
            res.append(pyaiger_to_spot_dict[token])
        else:
            if not token.startswith('var') or len(token) != 5:
                raise ValueError('Expected varXX')
            n = int(token[3:])
            if n >= 26:
                raise ValueError()
            res.append(chr(n + 97))
    return res


## Main things
def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num-aps', default=5, type=int)
    parser.add_argument('--num-examples', metavar='N', type=int, help='Number of examples to generate.', required=True)
    parser.add_argument('--target-directory', type=str,
                        help='Directory to which to write the tfexamples and the vocabulary.', required=True)
    parser.add_argument('--max-size', type=int, help='Maximum size of the formulas to generate', required=True)
    parser.add_argument('--alpha', type=float, default=0.0)
    args = parser.parse_args()

    if args.num_aps > 26:
        raise ValueError("Cannot generate more than 26 APs")
    return args

# def build_example(min_model, formula_obj, in_vocab, out_vocab):
#     assignments_spot = pyaiger_to_spot(min_model.split(' '))
#     assignments_pretty = ''
#     i = iter(assignments_spot)
#     for var in i:
#         val = next(i)
#         assignments_pretty += var+ '=' + val + ' '
#     a_enc = out_vocab.encode(assignments_spot, prepend_start_token=False)

#     polish_spot = formula_obj.to_str('network-polish', spacing='all ops')
#     f_enc = in_vocab.encode(polish_spot.split(' '))

#     features = {'formula_infix': tf_converter.string_feature(formula_obj.to_str('spot')), 'formula_polish_tokens' : tf_converter.int_feature(f_enc), 
#             'minimized_pretty' : tf_converter.string_feature(assignments_pretty), 'minimized_tokens' : tf_converter.int_feature(a_enc)}
#     example = tf.train.Example(features=tf.train.Features(feature=features))
#     return example

def split_and_write(examples, args, seed, gate):
    total_samples = len(examples)
    print("Generated a total of", total_samples, "examples")
    random.Random(seed).shuffle(examples)
    res = {}
    train_frac = 0.8
    val_frac = 0.1
    res['train'] = examples[0: int(train_frac * total_samples)]
    res['val'] = examples[int(train_frac * total_samples)  : int((train_frac + val_frac) * total_samples)]
    res['test'] = examples[int((train_frac + val_frac) * total_samples):]

    folder = utils.dataset_name(args.num_aps, args.max_size, args.num_examples, polish=True)
    directory = os.path.join(args.target_directory, folder)
    os.makedirs(directory)

    for part in ['train', 'test', 'val']:
        examples = res[part]
        path = os.path.join(directory, part + '.txt')
        print('Writing {:d} samples into {}'.format(len(examples), path))
        # path = os.path.join(directory, part + '.tfrecord')
        # with tf.io.TFRecordWriter(path) as file_writer:
        #     for example in examples:
        #         file_writer.write(example.SerializeToString())
        with open(path, 'w') as f:
            for formula, ass in examples:
                f.write(formula + '\n')
                f.write(ass + '\n')

    gate.histogram(show=False, save_to=os.path.join(directory, 'distribution.png'))


def main():
    args = parse_args()
    aps = list(map(chr, range(97, 97 + args.num_aps)))
    seed = 42
    formula_generator = spot.randltl(aps, seed=seed, tree_size=(1, args.max_size+2),
                        ltl_priorities='false=1,true=1,not=1,F=0,G=0,X=0,equiv=0,implies=0,xor=0,R=0,U=0,W=0,M=0,and=1,or=1', simplify=0)

    gate = DistributionGate('formula size', 'uniform', (1, args.max_size), args.num_examples, start_calc_from=12, alpha=args.alpha)
    worker = utils.PersistentWorker()
    worker_calls = 0
    samples = []
    total_samples = 0
    while total_samples < args.num_examples and not gate.full():
        formula_str_spot = next(formula_generator).to_str('lbt')
        formula_obj = ltl_parser.ltl_formula(formula_str_spot, 'lbt')
        polish_pyaiger = spot_to_pyaiger(formula_obj.to_str('network-polish', spacing='all ops').split(' '))
        if not gate.gate(formula_obj):
            continue
        if worker_calls >= 10000:
            worker.terminate()
            worker_calls = 0
        finished, min_model = worker.call(generate_model, (polish_pyaiger, None), 60)
        worker_calls += 1
        assert finished
        if min_model is None:
            continue # no unsat
        gate.update(formula_obj)

        assignments_spot = ''.join(pyaiger_to_spot(min_model.split(' ')))
        formula_spot = formula_obj.to_str('network-polish')
        samples.append((formula_spot, assignments_spot))
        total_samples += 1
        if total_samples % 10000 == 0:
            print(f'{total_samples/args.num_examples*100:5.1f}% complete')
            sys.stdout.flush()
    try:
        split_and_write(samples, args, seed, gate)
    finally:
        worker.terminate()


if __name__ == "__main__":
    # execute only if run as a script
    main()
