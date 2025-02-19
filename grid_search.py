import numpy
import theano
import theano.sandbox.softsign
from copy import copy
import datetime
import itertools
import os

import argparse

from MADE.weights_initializer import WeightsInitializer


activation_functions = {
    "sigmoid": theano.tensor.nnet.sigmoid,
    "hinge": lambda x: theano.tensor.maximum(x, 0.0),
    "softplus": theano.tensor.nnet.softplus,
    "tanh": theano.tensor.tanh,
    "softsign": theano.sandbox.softsign.softsign
}

LEARNING_RATES = [1e-5]  # [1e-5, 1e-7, 1e-9]
DECREASE_CONSTANTS = [0.95]
MAX_EPOCHS = [-1]  # [-1, 100]
SHUFFLE_MASKS = [1, 8]  # [1, 8, -1]  # [1, 8, 16, 32, -1]  # -1 no limit
SHUFFLING_TYPES = ['Full']  # ['Once', 'Full', 'Ordering', 'Connectivity']
NB_SHUFFLE_PER_VALID = [300]  # only considered if suffle_mask = -1
BATCH_SIZE = [100]
LOOK_AHEAD = [20]  # [30]
PRE_TRAINING = [False]  # [False, True]
PRE_TRAINING_MAX_EPOCH = [0]
UPDATE_RULE = ['adadelta']  # ['None', 'adadelta', 'adagrad', 'rmsprop', 'adam', 'adam_paper']
DROPOUT_RATE = [0]
HIDDEN_SIZES = ["[500]", "[500,500]"]
RANDOM_SEED = [1234]
USE_COND_MASK = [False]  # [False, True]
DIRECT_INPUT_CONNECT = ['Output']  # ["None", "Output", "Full"]
DIRECT_OUTPUT_CONNECT = [False]  # [False, True]
HIDDEN_ACTIVATIONS = ['hinge']  # ['hinge', 'softplus']  # list(activation_functions.keys())
WEIGHTS_INITIALIZATIONS = ['Orthogonal']  # [WeightsInitializer.Orthogonal]
MASK_DISTRIBUTIONS = [0]

CLUSTER_SCRIPT_FILENAME = 'made' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CLUSTER_SCRIPT_TEMPLATE = """executable = {}
arguments = {}
error = {}.err
output = {}.out
log = {}.log
request_memory = {}
request_cpus = {}
request_GPUs = 1
queue
"""

N_CPUS = 10
MEMORY = 1024 * 10
# OUT_DIR = 'exp/debug_only_when_exo_10k/i_crf-hp/'
OUT_DIR = 'exp'
os.makedirs(OUT_DIR, exist_ok=True)
BID = 100


def stringify(x):
    if isinstance(x, (int, float, bool)):
        return str(x)
    elif isinstance(x, list):
        # return '[{}]'.format([stringify(e) for e in x])
        return str(x)
    elif isinstance(x, str):
        return x


CLUSTER_PYTHON_INTERPRETER = '/home/avergari/.local/bin/ipython'
PYTHON_STR = 'ipython'
# MADE_TRAIN_STR = 'THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 {python} -- trainMADE.py {dataset} {params}'
MADE_TRAIN_STR = ' {python} -- trainMADE.py {params}'


def grid_search(python_int,
                datasets,
                learning_rate_values,
                decrease_constant_values,
                max_epochs_values,
                shuffle_mask_values,
                shuffling_type_values,
                nb_shuffle_per_valid_values,
                batch_size_values,
                look_ahead_values,
                pre_training_values,
                pre_training_max_epoc_values,
                update_rule_values,
                dropout_rate_values,
                hidden_sizes_values,
                random_seed_values,
                use_cond_mask_values,
                direct_input_connect_values,
                direct_output_connect_values,
                hidden_activation_values,
                weights_initialization_values,
                mask_distribution_values):

    i = 0
    #
    # create a product
    configs = list(itertools.product(datasets,
                                     learning_rate_values,
                                     decrease_constant_values,
                                     max_epochs_values,
                                     shuffle_mask_values,
                                     shuffling_type_values,
                                     nb_shuffle_per_valid_values,
                                     batch_size_values,
                                     look_ahead_values,
                                     pre_training_values,
                                     pre_training_max_epoc_values,
                                     update_rule_values,
                                     dropout_rate_values,
                                     hidden_sizes_values,
                                     random_seed_values,
                                     use_cond_mask_values,
                                     direct_input_connect_values,
                                     direct_output_connect_values,
                                     hidden_activation_values,
                                     weights_initialization_values,
                                     mask_distribution_values))
    for p, prod in enumerate(configs):

        #
        # # unpacking
        # learning_rate, decrease_constant, \
        #     max_epochs, shuffle_mask, shuffling_type, \
        #     nb_shuffle_per_valid, batch_size, look_ahead, \
        #     pre_training, pre_training_max_epoc, update_rule, \
        #     dropout_rate, hidden_sizes, random_seed, use_cond_mask, \
        #     direct_input_connect, direct_output_connect, hidden_activation, \
        #     weight_initialization, mask_distribution = prod

        params_str = " ".join([str(pa) for pa in prod])
        train_str = MADE_TRAIN_STR.format(python=python_int,
                                          # dataset=dataset,
                                          params=params_str)
        print(train_str)
        #
        # calling trainMADE as a process
        i += 1

        cluster_script_name = '{}-{}-{}'.format(CLUSTER_SCRIPT_FILENAME, prod[0], p)
        cluster_cmds = copy(CLUSTER_SCRIPT_TEMPLATE)
        # cluster_args = cluster_args.replace('"', '\\"')
        cluster_args = '"{}"'.format(train_str)
        cluster_args = train_str.replace(PYTHON_STR, '')

        cluster_cmds = cluster_cmds.format(CLUSTER_PYTHON_INTERPRETER,
                                           cluster_args,
                                           cluster_script_name, cluster_script_name, cluster_script_name,
                                           MEMORY, N_CPUS)
        exp_file_path = os.path.join(OUT_DIR, '{}.sub'.format(cluster_script_name))
        with open(exp_file_path, 'w') as f:
            f.write(cluster_cmds)

    genexp_file_path = os.path.join(OUT_DIR, 'gen-exp.sh')
    with open(genexp_file_path, 'w') as f:
        for p in range(i):
            f.write('condor_submit_bid {} ./{}\n'.format(BID,
                                                         os.path.join(OUT_DIR, '{}-{}-{}.sub'.format(CLUSTER_SCRIPT_FILENAME, configs[p][0], p))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the MADE model.')

    parser.add_argument('dataset_name', nargs='+')
    parser.add_argument('--learning_rate', type=float, nargs='+', default=LEARNING_RATES)
    parser.add_argument('--decrease_constant', type=float, nargs='+', default=DECREASE_CONSTANTS)
    parser.add_argument('--max_epochs', type=int, nargs='+',  default=MAX_EPOCHS)
    parser.add_argument('--shuffle_mask', type=int, nargs='+',  default=SHUFFLE_MASKS)
    parser.add_argument('--shuffling_type', nargs='+',  default=SHUFFLING_TYPES)
    parser.add_argument(
        '--nb_shuffle_per_valid', type=int, nargs='+',  default=NB_SHUFFLE_PER_VALID)
    parser.add_argument('--batch_size', type=int, nargs='+',  default=BATCH_SIZE)
    parser.add_argument('--look_ahead', type=int, nargs='+',  default=LOOK_AHEAD)
    parser.add_argument('--pre_training', type=eval, choices=[False, True], nargs='+',
                        default=PRE_TRAINING)
    parser.add_argument('--pre_training_max_epoc', type=int, nargs='+',
                        default=PRE_TRAINING_MAX_EPOCH)
    parser.add_argument('--update_rule', nargs='+', default=UPDATE_RULE)
    parser.add_argument('--dropout_rate', type=float, nargs='+', default=DROPOUT_RATE)

    parser.add_argument('--hidden_sizes', type=str, nargs='+', default=HIDDEN_SIZES)
    parser.add_argument('--random_seed', type=int,  nargs='+', default=RANDOM_SEED)
    parser.add_argument('--use_cond_mask',  type=eval, choices=[False, True], nargs='+',
                        default=USE_COND_MASK)
    parser.add_argument('--direct_input_connect', nargs='+',  default=DIRECT_INPUT_CONNECT)
    parser.add_argument('--direct_output_connect',  nargs='+', type=eval, choices=[False, True],
                        default=DIRECT_OUTPUT_CONNECT)
    parser.add_argument('--hidden_activation', nargs='+', default=HIDDEN_ACTIVATIONS)
    parser.add_argument('--weights_initialization', nargs='+',  default=WEIGHTS_INITIALIZATIONS)
    parser.add_argument('--mask_distribution', type=float, nargs='+', default=MASK_DISTRIBUTIONS)

    args = parser.parse_args()

    grid_search(PYTHON_STR,
                args.dataset_name,
                args.learning_rate,
                args.decrease_constant,
                args.max_epochs,
                args.shuffle_mask,
                args.shuffling_type,
                args.nb_shuffle_per_valid,
                args.batch_size,
                args.look_ahead,
                args.pre_training,
                args.pre_training_max_epoc,
                args.update_rule,
                args.dropout_rate,
                args.hidden_sizes,
                args.random_seed,
                args.use_cond_mask,
                args.direct_input_connect,
                args.direct_output_connect,
                args.hidden_activation,
                args.weights_initialization,
                args.mask_distribution)
