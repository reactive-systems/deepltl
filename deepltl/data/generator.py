#!python3.6

# pylint: disable=line-too-long

from __future__ import generator_stop  # just to be safe with python 3.7
import os
import argparse
import random
import sys

import spot
from deepltl.data import aalta_wrapper
from deepltl.data import ltl_parser
from deepltl.utils import utils


def spot_get_trace(formula_str, simplify):
    spot_formula = spot.formula(formula_str)
    automaton = spot_formula.translate()
    automaton.merge_edges()
    acc_run = automaton.accepting_run()
    if acc_run is None:
        return False, None
    else:
        trace = spot.twa_word(acc_run)
        if simplify:
            trace.simplify()
        return True, str(trace)


SPOT_WORKER: utils.PersistentWorker = None  # have to initialize and end


def get_sat_and_trace(formula_str, tool, simplify, timeout):
    if tool == 'spot':
        finished, res = SPOT_WORKER.call(
            spot_get_trace, (formula_str, simplify), timeout)
        if not finished:
            return None, None  # timeout
        sat, trace = res
        assert sat is not None
        return sat, trace
    elif tool == 'aalta':
        if simplify:
            raise ValueError("Simplify not supported for aalta")
        return aalta_wrapper.sat_with_evidence(formula_str, timeout=timeout)
    else:
        raise ValueError("Unknown tool")


class DistributionGate():
    # interval: [a, b]
    def __init__(self, key, distribution, interval, total_num, **kwargs):
        # optional: start_calc_at together with alpha
        self.dist = {}
        self.targets = {}
        self.fulls = {}
        self.key = key
        self.interval = interval
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0
        bleft, bright = interval
        if key == 'formula size':
            self.bins = list(range(bleft, bright + 1))
            self.get_val = lambda x: x.size()
        else:
            raise ValueError()
        for b in self.bins:
            self.dist[b] = 0
        if distribution == 'uniform':
            if 'start_calc_from' in kwargs:
                start = kwargs['start_calc_from']
                self.enforced_bins = list(
                    filter(lambda x: x >= start, self.bins))
            else:
                self.enforced_bins = self.bins
            num_actual_bins = len(self.enforced_bins)
            for b in self.bins:
                self.targets[b] = total_num * \
                    (1 - self.alpha) / num_actual_bins
                self.fulls[b] = self.dist[b] >= self.targets[b]
        else:
            raise ValueError()

    def gate(self, formula_obj: ltl_parser.LTLFormula) -> bool:
        val = self.get_val(formula_obj)
        if val < self.interval[0] or val > self.interval[1]:  # not in range
            return False
        return not self.fulls[val]

    def update(self, formula_obj: ltl_parser.LTLFormula):
        val = self.get_val(formula_obj)
        if val >= self.interval[0] and val <= self.interval[1]:
            self.dist[val] += 1
            if self.dist[val] >= self.targets[val]:
                self.fulls[val] = True

    def histogram(self, show=True, save_to=None):
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(1)
        counts = [val for key, val in sorted(self.dist.items())]
        axis.bar(self.bins, counts, width=1,
                 color='#3071ff', edgecolor='white')
        axis.set_ylabel('number of items')
        axis.set_xlabel(self.key)
        axis.title.set_text('alpha = ' + str(self.alpha))
        if save_to is not None:
            figure.savefig(save_to)
        if show:
            plt.show()
        else:
            plt.close(figure)

    def full(self) -> bool:
        return all([self.fulls[eb] for eb in self.enforced_bins])


def generate_samples(num_aps, num_formulas, tree_size, seed, polish, simplify, train_frac, val_frac, unsat_frac, trace_generator, timeout, require_trace, alpha, **kwargs):
    if num_aps > 26:
        raise ValueError("Cannot generate more than 26 APs")
    aps = list(map(chr, range(97, 97 + num_aps)))

    if isinstance(tree_size, int):
        tree_size = (1, tree_size)
    formula_generator = spot.randltl(aps, seed=seed, tree_size=tree_size,
                                     ltl_priorities='false=1,true=1,not=1,F=0,G=0,X=1,equiv=0,implies=0,xor=0,R=0,U=1,W=0,M=0,and=1,or=0', simplify=0)

    tictoc = utils.TicToc()
    dist_gate = DistributionGate(
        'formula size', 'uniform', tree_size, num_formulas, start_calc_from=10, alpha=alpha)
    global SPOT_WORKER
    SPOT_WORKER = utils.PersistentWorker()

    # generate samples
    print('Generating samples...')
    sat_only = unsat_frac == 0.0
    samples = []
    sat_samples = 0
    unsat_samples = 0
    total_samples = 0
    while total_samples < num_formulas and not dist_gate.full():
        tictoc.tic()
        formula_spot = next(formula_generator)
        tictoc.toc('formula generation')
        formula_str = formula_spot.to_str()
        formula_obj = ltl_parser.ltl_formula(formula_str, 'spot')
        if not dist_gate.gate(formula_obj):  # formula doesn't fit distribution
            continue
        # add some spaces and parenthesis to be safe for aalta
        formula_spaced = formula_obj.to_str(
            'spot', spacing='all ops', full_parens=True)
        tictoc.tic()
        is_sat, trace_str = get_sat_and_trace(
            formula_spaced, trace_generator, simplify, timeout)
        tictoc.toc('trace generation')

        if is_sat is None:  # due to timeout
            print('Trace generation timed out ({:d}s) for formula {}'.format(
                int(timeout), formula_obj.to_str('spot')))
            if require_trace:
                continue
            else:  # no trace required
                trace_str = '-'
                dist_gate.update(formula_obj)
        elif not is_sat and sat_only:
            continue
        elif not is_sat and not sat_only:
            if unsat_samples >= unsat_frac * num_formulas:
                continue
            else:  # more unsat samples needed
                trace_str = '{0}'
                dist_gate.update(formula_obj)
                unsat_samples += 1
        else:  # is_sat
            if '0' in trace_str:
                print('Bug in spot! (trace containing 0):\nFormula: {}\nTrace: {}\n'.format(
                    formula_obj.to_str('spot'), trace_str))
                continue
            assert unsat_samples < unsat_frac * \
                num_formulas or not ('0' in trace_str and not sat_only)
            if sat_samples >= (1 - unsat_frac) * num_formulas:
                continue
            else:  # more sat samples needed
                trace_str = ltl_parser.ltl_trace(trace_str, 'spot').to_str(
                    'network-' + ('polish' if polish else 'infix'))
                dist_gate.update(formula_obj)
                sat_samples += 1

        formula_str = formula_obj.to_str(
            'network-' + ('polish' if polish else 'infix'))
        samples.append((formula_str, trace_str))
        if total_samples % (num_formulas // 10) == 0 and total_samples > 0:
            print("%d/%d" % (total_samples, num_formulas))
        total_samples += 1
        sys.stdout.flush()
    # dist_gate.histogram(show=False, save_to='dist_nf{}_ts{:d}-{:d}.png'.format(utils.abbrev_count(num_formulas), tree_size[0], tree_size[1]))      # For distribution analysis
    # tictoc.histogram(show=False, save_to='timing_nf{}_ts{:d}-{:d}.png'.format(utils.abbrev_count(num_formulas), tree_size[0], tree_size[1]))         # For timing analysis
    print('Generated {:d} samples, {:d} requested'.format(
        total_samples, num_formulas))
    SPOT_WORKER.terminate()

    # shuffle and split samples
    random.Random(seed).shuffle(samples)
    res = {}
    res['train'] = samples[0: int(train_frac * total_samples)]
    res['val'] = samples[int(train_frac * total_samples)                         : int((train_frac + val_frac) * total_samples)]
    res['test'] = samples[int((train_frac + val_frac) * total_samples):]
    return res


def run():
    parser = argparse.ArgumentParser(
        description='Randomly generates LTL formulas with a corresponding trace.')
    parser.add_argument('--num-aps', '-na', type=int, default=5)
    parser.add_argument('--num-formulas', '-nf', type=int, default=1000)
    parser.add_argument('--tree-size', '-ts', type=str, default='15', metavar='MAX_TREE_SIZE',
                        help="Maximum tree size of generated formulas. Range can be specified as 'MIN-MAX'; default minimum is 1")
    parser.add_argument('--output-dir', '-od', type=str, default="data")
    parser.add_argument('--seed', type=int, default=42)
    infix_or_polish = parser.add_mutually_exclusive_group()
    infix_or_polish.add_argument('--polish', dest='polish', action='store_true',
                                 default=True, help='write formulas and traces in polish notation; default')
    infix_or_polish.add_argument('--infix', dest='polish', action='store_false',
                                 default=True, help='write formulas and traces in infix notation')
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--unsat-frac', type=float, default=0.0)
    parser.add_argument('--trace-generator', type=str, choices=[
                        'spot', 'aalta'], default='spot', help='which tool to get a trace (or unsat) from; default spot')
    parser.add_argument('--timeout', type=float, default=10,
                        help='time in seconds to wait for the trace generator to return, if expired kill and continue with next formula')
    require_trace = parser.add_mutually_exclusive_group()
    require_trace.add_argument('--require-trace', dest='require_trace', action='store_true', default=True,
                               help='require a trace to be found for each formula (useful for training/testing set); default')
    require_trace.add_argument('--allow-no-trace', dest='require_trace', action='store_false', default=True,
                               help='allow formulas without a corresponding trace found (useful for further evaluation)')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Distribution parameter')
    parser.add_argument('--name-prefix', help="Name to prefix the dataset name with")
    args = parser.parse_args()

    tree_size = args.tree_size.split('-')
    if len(tree_size) == 1:
        tree_size = int(tree_size[0])
    else:
        tree_size = (int(tree_size[0]), int(tree_size[1]))
    args_dict = vars(args)
    args_dict['tree_size'] = tree_size
    res = generate_samples(**args_dict)

    folder = utils.dataset_name(**args_dict)
    directory = os.path.join(args.output_dir, folder)
    os.makedirs(directory)

    # write samples to files
    for part in ['train', 'test', 'val']:
        if part in res:
            samples = res[part]
            path = os.path.join(directory, part + '.txt')
            print('Writing {:d} samples into {}'.format(len(samples), path))
            with open(path, 'w') as f:
                for formula_str, trace_str in samples:
                    f.write(formula_str + '\n' + trace_str + '\n')


if __name__ == '__main__':
    run()
