# pylint: disable = line-too-long
import os
import os.path as path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # reduce TF verbosity
import tensorflow as tf
import urllib.request

from deepltl.train.common import *
from deepltl.optimization import lr_schedules
from deepltl.models import transformer
from deepltl.data import vocabulary
from deepltl.data import datasets

def download_dataset(dataset_name, problem, splits, data_dir):
    if not path.isdir(data_dir):
        os.mkdir(data_dir)

    url = 'https://storage.googleapis.com/deepltl_data/data/'

    if problem == 'ltl':
        url += 'ltl_traces/'
        dataset_dir = path.join(data_dir, 'ltl')
    if problem == 'prop':
        url += 'sat/'
        dataset_dir = path.join(data_dir, 'prop')
    
    if not path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    url += dataset_name + '/'

    dataset_dir = path.join(dataset_dir, dataset_name)
    if not path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    #check if splits arleady exist
    for split in splits:
        split_dir = path.join(dataset_dir, split + '.txt')
        if not path.isfile(split_dir):
            print(f'Downloading {split} ...')
            urllib.request.urlretrieve(url + split + '.txt', split_dir)


def run():
    # Argument parsing
    parser = argparser()
    # add specific arguments
    parser.add_argument('--problem', type=str, default='ltl', help='available problems: ltl, prop')
    parser.add_argument('--d-embed-enc', type=int, default=128)
    parser.add_argument('--d-embed-dec', type=int, default=None)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--ff-activation', default='relu')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--tree-pos-enc', action='store_true', default=False, help='use tree positional encoding')
    params = parser.parse_args()
    setup(**vars(params))

    aps = ['a', 'b', 'c', 'd', 'e']
    consts = ['0', '1']

    if params.problem != 'ltl' and params.problem != 'prop':
        sys.exit(f'{params.problem} is not a valid problem\n')

    # Dataset specification
    if params.ds_name is None:
        sys.exit('No dataset specified\n')
    else:
        if params.ds_name == 'ltl-35' or params.ds_name == 'prop-35':
            params.max_encode_length = 37
            dataset_name = 'na-5-ts-35-nf-1m-lbt-sat'
        elif params.ds_name == 'ltl-50-test' or params.ds_name == 'prop-50-test':
            params.max_encode_length = 52
            if not params.test: # train mode
                sys.exit(f'Dataset {params.ds_name} can only be used in test mode\n')
            dataset_name = 'na-5-ts-35-50-nf-20k-lbt-sat'
        elif params.ds_name == 'prop-60-no-derived':
            params.max_encode_length = 62
            aps += ['f', 'g', 'h', 'i', 'j']
            dataset_name = 'lessops_na-10-ts-60-nf-1m-lbt-sat'
        else:
            sys.exit(f'{params.ds_name} is not a valid dataset\n')

    if not params.test: # train mode
        download_dataset(dataset_name, params.problem, ['train', 'val', 'test'], params.data_dir)
    else: # test only
        download_dataset(dataset_name, params.problem, ['test'], params.data_dir)
    
    if params.problem == 'ltl':
        input_vocab = vocabulary.LTLVocabulary(aps=aps, consts=consts, ops=['U', 'X', '!', '&'], eos=not params.tree_pos_enc)
        target_vocab = vocabulary.TraceVocabulary(aps=aps, consts=consts, ops=['&', '|', '!'])
        dataset = datasets.LTLTracesDataset(dataset_name, input_vocab, target_vocab, data_dir=path.join(params.data_dir, 'ltl'))
        params.max_decode_length = 64
    elif params.problem == 'prop':
        input_vocab = vocabulary.LTLVocabulary(aps, consts, ['!', '&', '|', '<->', 'xor'], eos=not params.tree_pos_enc)
        target_vocab = vocabulary.TraceVocabulary(aps, consts, [], special=[])
        dataset = datasets.BooleanSatDataset(dataset_name, data_dir=path.join(params.data_dir, 'prop'), formula_vocab=input_vocab, assignment_vocab=target_vocab)
        if params.ds_name == 'prop-60-no-derived':
            params.max_decode_length = 22
        else:
            params.max_decode_length = 12

    params.input_vocab_size = input_vocab.vocab_size()
    params.input_pad_id = input_vocab.pad_id
    params.target_vocab_size = target_vocab.vocab_size()
    params.target_start_id = target_vocab.start_id
    params.target_eos_id = target_vocab.eos_id
    params.target_pad_id = target_vocab.pad_id
    params.dtype = tf.float32

    if params.d_embed_dec is None:
        params.d_embed_dec = params.d_embed_enc
    print('Specified dimension of encoder embedding:', params.d_embed_enc)
    params.d_embed_enc -= params.d_embed_enc % params.num_heads  # round down
    print('Specified dimension of decoder embedding:', params.d_embed_dec)
    params.d_embed_dec -= params.d_embed_dec % params.num_heads  # round down
    print('Parameters:')
    for key, val in vars(params).items():
        print('{:25} : {}'.format(key, val))

    if not params.test: # train mode
        splits = ['train', 'val', 'test']
        if params.problem == 'ltl':
            train_dataset, val_dataset, test_dataset = dataset.get_dataset(splits, max_length_formula=params.max_encode_length - 2, max_length_trace=params.max_decode_length - 2, prepend_start_token=False, tree_pos_enc=params.tree_pos_enc)
        if params.problem == 'prop':
            train_dataset, val_dataset, test_dataset = dataset.get_dataset(splits=splits, tree_pos_enc=params.tree_pos_enc)
        train_dataset = prepare_dataset_no_tf(train_dataset, params.batch_size, params.d_embed_enc, shuffle=True, pos_enc=params.tree_pos_enc)
        val_dataset = prepare_dataset_no_tf(val_dataset, params.batch_size, params.d_embed_enc, shuffle=False,  pos_enc=params.tree_pos_enc)
    else:  # test mode
        splits = ['test']
        if params.problem == 'ltl':
            test_dataset, = dataset.get_dataset(['test'], max_length_formula=params.max_encode_length - 2, max_length_trace=params.max_decode_length - 2, prepend_start_token=False, tree_pos_enc=params.tree_pos_enc)
        if params.problem == 'prop':
            test_dataset, = dataset.get_dataset(splits=['test'], tree_pos_enc=params.tree_pos_enc)

    if not params.test: # train mode
        # Model & Training specification
        learning_rate = lr_schedules.TransformerSchedule(params.d_embed_enc, warmup_steps=params.warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model = transformer.create_model(vars(params), training=True, custom_pos_enc=params.tree_pos_enc)
        latest_checkpoint = last_checkpoint(**vars(params))
        if latest_checkpoint:
            model.load_weights(latest_checkpoint).expect_partial()
            print(f'Loaded weights from checkpoint {latest_checkpoint}')

        callbacks = [checkpoint_callback(save_weights_only=True, save_best_only=False, **vars(params)),
                     tensorboard_callback(**vars(params)),
                     tf.keras.callbacks.EarlyStopping('val_accuracy', patience=4, restore_best_weights=True)]
        # Train!
        log_params(**vars(params))
        model.compile(optimizer=optimizer)
        try:
            model.fit(train_dataset, epochs=params.epochs, validation_data=val_dataset, validation_freq=1, callbacks=callbacks, initial_epoch=params.initial_epoch, verbose=1, shuffle=False)
        except Exception as e:
            print('---- Exception occurred during training ----\n' + str(e))
    else:  # test mode
        prediction_model = transformer.create_model(vars(params), training=False, custom_pos_enc=params.tree_pos_enc)
        latest_checkpoint = last_checkpoint(**vars(params))
        if latest_checkpoint:
            prediction_model.load_weights(latest_checkpoint).expect_partial()
            print(f'Loaded weights from checkpoint {latest_checkpoint}')
        else:
            sys.exit('Could not load weights from checkpoint')
        sys.stdout.flush()

        padded_shapes = ([None], [None, params.d_embed_enc], [None]) if params.tree_pos_enc else ([None], [None])
        test_dataset = test_dataset.shuffle(100000, seed=42).take(100).padded_batch(params.batch_size, padded_shapes=padded_shapes)

        if params.problem == 'ltl':
            if params.tree_pos_enc:
                def pred_fn(x, pe):
                    output, _ = prediction_model([x, pe], training=False)
                    return output
            else:
                def pred_fn(x):
                    output, _ = prediction_model(x, training=False)
                    return output
            test_and_analyze_ltl(pred_fn, test_dataset, input_vocab, target_vocab, log_name='test.log', **vars(params))

        if params.problem == 'prop':
            test_and_analyze_sat(prediction_model, test_dataset, input_vocab, target_vocab, log_name='test.log', **vars(params))


if __name__ == '__main__':
    run()
