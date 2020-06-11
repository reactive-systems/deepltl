from timeit import default_timer as timer
import multiprocessing as mp
import math

class TicToc():
    def __init__(self):
        self.t = None
        self.results = {}

    def tic(self):
        self.t = timer()

    def toc(self, name):
        if self.t is None:
            raise RuntimeError('Timer not started')
        diff = timer() - self.t
        self.t = None
        if name in self.results:
            self.results[name].append(diff)
        else:
            self.results[name] = [diff]

    def histogram(self, show=True, save_to=None, figsize=None):
        import matplotlib.pyplot as plt

        num_subplots = len(self.results)
        if figsize is None:
            figsize = (num_subplots * 5, 5)
        figure, axes = plt.subplots(1, num_subplots, figsize=figsize)
        if num_subplots == 1:
            axes = [axes]
        for idx, (name, vals) in enumerate(self.results.items()):
            axes[idx].hist(vals)
            axes[idx].set_xlabel('time / s')
            axes[idx].title.set_text(name)
            axes[idx].set_yscale('log', nonposy='clip')
        if save_to is not None:
            figure.savefig(save_to)
        if show:
            plt.show()
        else:
            plt.close(figure)


def _wrapped_target(pipe_conn: 'mp.connection.Connection'):
    target, args = pipe_conn.recv()
    res = target(*args)
    pipe_conn.send(res)

def run_with_timeout(target, timeout, args=()):
    """Wraps target function into a subprocess where arguments and result are transferred via pipes. Terminates process after timeout seconds. Returns if finished by its own and function result"""
    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=_wrapped_target, args=(child_conn,))
    proc.start()
    parent_conn.send((target, args))
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, None
    else:
        res = parent_conn.recv()
        return True, res


def _wrapped_target_persistent(pipe_conn: 'mp.connection.Connection'):
    while True:
        target, args = pipe_conn.recv()
        res = target(*args)
        pipe_conn.send(res)

class PersistentWorker():
    def __init__(self):
        self.process = None
        self.connection = None

    def call(self, target, args, timeout):
        if self.process is None:
            self.connection, child_conn = mp.Pipe()
            self.process = mp.Process(target=_wrapped_target_persistent, args=(child_conn,))
            self.process.start()

        self.connection.send((target, args))
        if not self.connection.poll(timeout):
            self.process.terminate()
            self.process.join()
            self.process = None
            return False, None
        else:
            res = self.connection.recv()
            return True, res

    def terminate(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.process = None
            self.connection = None


def abbrev_count(count):
    log_count = math.floor(math.log10(count))
    k_exponent = math.floor(log_count / 3)
    suffixes = ['', 'k', 'm']
    return '{:g}{}'.format(count / 10**(k_exponent*3), suffixes[k_exponent])


def dataset_name(num_aps, tree_size, num_formulas, polish=True, unsat_frac=0.0, simplify=False, require_trace=True, name_prefix=None, **kwargs):
    folder = name_prefix + '-' if name_prefix is not None else ''

    if isinstance(tree_size, int):
        tree_size = str(tree_size)
    else:
        tree_size = str(tree_size[0]) + '-' + str(tree_size[1])
    folder_substrs = ['na', str(num_aps), 'ts', tree_size, 'nf']
    folder_substrs.append(abbrev_count(num_formulas))
    folder += '-'.join(folder_substrs)
    if polish:
        folder += '-lbt'
    if unsat_frac <= 0.0:
        folder += '-sat'
    else:
        folder += '-unsat-' + str(unsat_frac)
    if simplify:
        folder += '-simpl'
    if not require_trace:
        folder += '-open'
    return folder
