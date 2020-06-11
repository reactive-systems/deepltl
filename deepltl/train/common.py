from argparse import ArgumentParser
import subprocess
import os.path as path
import sys

import tensorflow as tf
import numpy as np
from deepltl.data.datasets import teacher_forcing_map_fn, decoding_required_map_fn
from deepltl.data.vocabulary import CharacterVocabulary
from deepltl.data import ltl_parser


def prepare_datasets_tf(datasets, batch_size, token_start, seq_len_in=None, seq_len_out=None, sample_counts: dict = {}):
    seq_len_out = seq_len_out + 1 if seq_len_out is not None else None  # for <start> token
    sample_counts_ = {'train': None, 'val': None, 'test': None}
    sample_counts_.update(sample_counts)
    forged = {}
    for name, ds in datasets.items():
        # get data
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)  # generate data asynchronously
        # limit
        if sample_counts[name] is not None:
            ds = ds.take(sample_counts[name])  # limit samples
        # map for fit / evaluate
        if name in ['train', 'val']:  # map using teacher forcing
            ds = ds.map(teacher_forcing_map_fn(token_start))
        else:  # map for decoding
            #ds = ds.map(decoding_required_map_fn(token_start))
            # currently, do not map anything, do decoding later by hand
            pass
        # cache
        ds = ds.cache()
        # shuffle
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
        # batch
        if name in ['train', 'val']:  # TODO: for all
            ds = ds.padded_batch(batch_size, padded_shapes=(([seq_len_in], [seq_len_out]), [seq_len_out]), drop_remainder=True)
        # save
        forged[name] = ds
    return forged


def prepare_dataset_no_tf(dataset, batch_size, d_embedding, shuffle=True, pos_enc=False):
    def shape_dataset(x, y):
        return ((x, y), y)

    def shape_pos_enc_dataset(x, y, z):
        return ((x, y, z), z)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(shape_pos_enc_dataset if pos_enc else shape_dataset)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(100000, reshuffle_each_iteration=True)
    padded_shapes = (([None], [None, d_embedding], [None]), [None]) if pos_enc else (([None], [None]), [None])
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)

    return dataset


def argparser():
    parser = ArgumentParser()
    # Meta
    parser.add_argument('--run-name', default='default', help='name of this run, to better find produced data later')
    parser.add_argument('--job-dir', default='runs', help='general job directory to save produced data into')
    parser.add_argument('--data-dir', default='data', help='directory of datasets')
    parser.add_argument('--ds-name', default='ltl-35', help='Name of the dataset to use')
    do_test = parser.add_mutually_exclusive_group()
    do_test.add_argument('--train', dest='test', action='store_false', default=False, help='Run in training mode, do not perform testing; default')
    do_test.add_argument('--test', dest='test', action='store_true', default=False, help='Run in testing mode, do not train')
    parser.add_argument('--binary-path', default=None, help='Path to binaries, current: aalta')

    # Typical Hyperparameters
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--initial-epoch', type=int, default=0, help='used to track the epoch number correctly when resuming training')
    parser.add_argument('--training-samples', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beam-size', type=int, default=2)

    return parser


def setup(binary_path, **kwargs):
    # GPU stuff
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('GPUs', gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Get binaries
    filenames = ['aalta']
    if binary_path is not None:
        for filename in filenames:
            try:
                tf.io.gfile.makedirs('bin')
                tf.io.gfile.copy(path.join(binary_path, filename), path.join('bin', filename))
            except tf.errors.AlreadyExistsError:
                pass


def checkpoint_path(job_dir, run_name, **kwargs):
    return path.join(job_dir, run_name, 'checkpoints')


def checkpoint_callback(job_dir, run_name, save_weights_only=True, save_best_only=False, **kwargs):
    if save_best_only:
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'best'))  # save best only
    else:
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'cp_')) + 'ep{epoch:02d}_vl{val_loss:.3f}'  # save per epoch
    return tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=save_weights_only, save_best_only=save_best_only)


def tensorboard_callback(job_dir, run_name, **kwargs):
    log_dir = str(path.join(job_dir, run_name))
    return tf.keras.callbacks.TensorBoard(log_dir)


def last_checkpoint(job_dir, run_name, **kwargs):
    return tf.train.latest_checkpoint(checkpoint_path(job_dir, run_name))


def test_and_analyze_ltl(pred_fn, dataset, in_vocab=None, out_vocab=None, plot_name='test_results', log_name=None, **kwargs):
    plotdir = path.join(kwargs['job_dir'], kwargs['run_name'])
    tf.io.gfile.makedirs(plotdir)
    proc_args = ['-f', '-', '-t', '-', '-r', '-', '--per-size', '--save-analysis', 'tmp_test_results', '--validator', 'spot', '--log-level', '3']
    if log_name is not None:
        proc_args.extend(['-l', path.join(plotdir, log_name)])
    proc = subprocess.Popen(['python3', '-m', 'deepltl.data.trace_check'] + proc_args,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1000000)
    try:
        for x in dataset:
            if kwargs['tree_pos_enc']:
                data, pe, label = x
                pred = pred_fn(data, pe)
            else:
                data, label = x
                pred = pred_fn(data)
            if len(pred.shape) == 1:
                pred = np.expand_dims(pred, axis=0)
                data = tf.expand_dims(data, axis=0)
                label = tf.expand_dims(label, axis=0)
            for i in range(pred.shape[0]):
                label_decoded = out_vocab.decode(list(label[i, :]))
                if not label_decoded:
                    label_decoded = '-'
                step_in = in_vocab.decode(list(data[i, :])) + '\n' + out_vocab.decode(list(pred[i, :])) + '\n' + label_decoded + '\n'
                proc.stdin.write(step_in)
                proc.stdin.flush()
    except BrokenPipeError:
        sys.exit('Pipe to trace checker broke. output:' + proc.communicate()[0])
    sys.stdout.flush()
    proc_out, _ = proc.communicate()
    tf.io.gfile.copy('tmp_test_results.png', path.join(plotdir, plot_name + '.png'), overwrite=True)
    tf.io.gfile.remove('tmp_test_results.png')
    tf.io.gfile.copy('tmp_test_results.svg', path.join(plotdir, plot_name + '.svg'), overwrite=True)
    tf.io.gfile.remove('tmp_test_results.svg')
    return proc_out


def get_ass(lst):
    if len(lst) % 2 != 0:
        raise ValueError('length of assignments not even')
    ass_it = iter(lst)
    ass_dict = {}
    for var in ass_it:
        val = next(ass_it)
        if val == 'True' or val == '1':
            ass_dict[var] = True
        elif val == 'False' or val == '0':
            ass_dict[var] = False
        else:
            raise ValueError('assignment var not True or False')
    s = [f'{var}={val}' for (var, val) in ass_dict.items()]
    return ass_dict, ' '.join(s)


def test_and_analyze_sat(pred_model, dataset, in_vocab, out_vocab, log_name, **kwargs):
    from deepltl.data.sat_generator import spot_to_pyaiger, is_model

    logdir = path.join(kwargs['job_dir'], kwargs['run_name'])
    tf.io.gfile.makedirs(logdir)
    with open(path.join(logdir, log_name), 'w') as log_file:
        res = {'invalid': 0, 'incorrect': 0, 'syn_correct': 0, 'sem_correct': 0}
        for x in dataset:
            if kwargs['tree_pos_enc']:
                data, pe, label_ = x
                prediction, _ = pred_model([data, pe], training=False)
            else:
                data, label_ = x
                prediction, _ = pred_model(data, training=False)
            for i in range(prediction.shape[0]):
                formula = in_vocab.decode(list(data[i, :]), as_list=True)
                pred = out_vocab.decode(list(prediction[i, :]), as_list=True)
                label = out_vocab.decode(list(label_[i, :]), as_list=True)
                formula_obj = ltl_parser.ltl_formula(''.join(formula), 'network-polish')
                formula_str = formula_obj.to_str('spot')
                _, pretty_label_ass = get_ass(label)
                try:
                    _, pretty_ass = get_ass(pred)
                except ValueError as e:
                    res['invalid'] += 1
                    msg = f"INVALID ({str(e)})\nFormula: {formula_str}\nPred:     {' '.join(pred)}\nLabel:    {pretty_label_ass}\n"
                    log_file.write(msg)
                    continue
                if pred == label:
                    res['syn_correct'] += 1
                    msg = f"SYNTACTICALLY CORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:    {pretty_label_ass}\n"
                    # log_file.write(msg)
                    continue

                # semantic checking
                formula_pyaiger = spot_to_pyaiger(formula)
                ass_pyaiger = spot_to_pyaiger(pred)
                pyaiger_ass_dict, _ = get_ass(ass_pyaiger)
                # print(f'f: {formula_pyaiger}, ass: {pyaiger_ass_dict}')
                try:
                    holds = is_model(formula_pyaiger, pyaiger_ass_dict)
                except KeyError as e:
                    res['incorrect'] += 1
                    msg = f"INCORRECT (var {str(e)} not in formula)\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:  {pretty_label_ass}\n"
                    log_file.write(msg)
                    continue
                if holds:
                    res['sem_correct'] += 1
                    msg = f"SEMANTICALLY CORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:  {pretty_label_ass}\n"
                    log_file.write(msg)
                else:
                    res['incorrect'] += 1
                    msg = f"INCORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:   {pretty_label_ass}\n"
                    log_file.write(msg)

        total = sum(res.values())
        correct = res['syn_correct'] + res['sem_correct']
        msg = (f"Correct: {correct/total*100:.1f}%, {correct} out of {total}\nSyntactically correct: {res['syn_correct']/total*100:.1f}%\nSemantically correct: {res['sem_correct']/total*100:.1f}%\n"
               f"Incorrect: {res['incorrect']/total*100:.1f}%\nInvalid: {res['invalid']/total*100:.1f}%\n")
        log_file.write(msg)
        print(msg, end='')


def decode_to_file(model, dataset, decoder, out_vocab, filename, **kwargs):
    with tf.io.gfile.GFile(filename, 'w') as outfile:
        for data, label in dataset:
            pred = model.predict(data, decoder)
            outfile.write(out_vocab.decode(list(pred)) + '\n')


def log_params(job_dir, run_name, **kwargs):
    logdir = path.join(job_dir, run_name)
    tf.io.gfile.makedirs(logdir)
    with tf.io.gfile.GFile(path.join(logdir, 'params.txt'), 'w') as f:
        for key, val in kwargs.items():
            f.write('{:25} : {}\n'.format(key, val))
