#!/usr/bin/python -u

from __future__ import division
from __future__ import print_function
import sys
import os
import time as t
try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import numpy as np
from numpy.testing import assert_array_almost_equal
import theano
import theano.tensor.nnet.nnet
from scipy.misc import logsumexp
from MADE.weights_initializer import WeightsInitializer
from MADE.made import MADE
from dataset import Dataset
import utils

import pickle
import gzip


def get_mean_error_and_std(model, error_fnc, set_size, shuffle_mask, shuffling_type, nb_shuffle=1):
    if shuffle_mask > 0:
        nb_shuffle = shuffle_mask + 1

    if not shuffle_mask:
        nb_shuffle = 1

    log_probs = np.zeros((set_size, nb_shuffle))

    for i in range(nb_shuffle):
        if shuffle_mask:
            model.shuffle(shuffling_type)
        log_probs[:, i] = error_fnc(False)

    losses = np.log(log_probs.shape[1]) - logsumexp(log_probs, axis=1)
    return round(losses.mean(), 6), round(losses.std() / np.sqrt(losses.shape[0]), 6)


def get_mean_error_and_std_final(model, error_fnc, set_size, shuffle_mask, shuffling_type, nb_shuffle=1):
    if shuffle_mask > 0:
        nb_shuffle = shuffle_mask + 1

    if not shuffle_mask:
        nb_shuffle = 1

    log_probs = np.zeros((set_size, nb_shuffle))

    batch_size = 1000
    nb_iterations = int(np.ceil(set_size / batch_size))

    for i in range(nb_shuffle):
        if shuffle_mask:
            model.shuffle(shuffling_type)

        for index in range(nb_iterations):
            start = index * batch_size
            log_probs[:, i][start:start + batch_size] = error_fnc(index, False)

    losses = np.log(log_probs.shape[1]) - logsumexp(log_probs, axis=1)
    return round(losses.mean(), 6), round(losses.std() / np.sqrt(losses.shape[0]), 6)


def train_model(model, dataset, look_ahead, shuffle_mask, nb_shuffle_per_valid, max_epochs, batch_size, shuffling_type, save_model_path=None, trainer_status=None):
    start_training_time = t.time()

    if trainer_status is None:
        trainer_status = {
            "nb_shuffles": 0,
            "best_valid_error": np.inf,
            "best_epoch": 0,
            "epoch": 0,
            "nb_of_epocs_without_improvement": 0
        }

    # Always do a first shuffle so that the natural order does not gives us an edge
    model.shuffle("Full")
    # Reseting the mask to where they were when saved
    for i in range(trainer_status["nb_shuffles"]):
        model.shuffle(shuffling_type)

    print('\n### Training MADE ###')
    while(trainer_status["epoch"] < max_epochs and trainer_status["nb_of_epocs_without_improvement"] < look_ahead):
        trainer_status["epoch"] += 1

        print('Epoch {0} (Batch Size {1})'.format(trainer_status["epoch"], batch_size))
        print('\tTraining   ...', end=' ')
        start_time = t.time()
        nb_iterations = int(np.ceil(dataset['train']['length'] / batch_size))
        train_err = 0
        for index in range(nb_iterations):
            train_err += model.learn(index, True)

            if shuffle_mask:
                if trainer_status["nb_shuffles"] == shuffle_mask:
                    trainer_status["nb_shuffles"] = 0
                    model.reset(shuffling_type)
                else:
                    model.shuffle(shuffling_type)
                    trainer_status["nb_shuffles"] += 1

        print(utils.get_done_text(start_time),
              " avg NLL: {0:.6f}".format(train_err / nb_iterations))

        print('\tValidating ...', end=' ')
        start_time = t.time()
        if shuffle_mask > 0:
            model.reset(shuffling_type)
        valid_err, valid_err_std = get_mean_error_and_std(model, model.valid_log_prob, dataset['valid'][
                                                          'length'], shuffle_mask, shuffling_type, nb_shuffle_per_valid)
        if shuffle_mask > 0:
            model.reset(shuffling_type, trainer_status["nb_shuffles"])
        print(utils.get_done_text(start_time), " NLL: {0:.6f}".format(valid_err))

        if valid_err < trainer_status["best_valid_error"]:
            trainer_status["best_valid_error"] = valid_err
            trainer_status["best_epoch"] = trainer_status["epoch"]
            trainer_status["nb_of_epocs_without_improvement"] = 0
            # Save best model
            if save_model_path is not None:
                save_model_params(model, save_model_path)
                utils.save_dict_to_json_file(
                    os.path.join(save_model_path, "trainer_status"), trainer_status)
        else:
            trainer_status["nb_of_epocs_without_improvement"] += 1

    print("### Training", utils.get_done_text(start_training_time), "###")
    total_train_time = t.time() - start_training_time
    return trainer_status["best_epoch"], total_train_time


def build_model(dataset, trainingparams, hyperparams, hidden_sizes):
    print('\n### Initializing MADE ... ', end=' ')
    start_time = t.time()
    model = MADE(dataset,
                 learning_rate=trainingparams['learning_rate'],
                 decrease_constant=trainingparams['decrease_constant'],
                 hidden_sizes=hidden_sizes,
                 random_seed=hyperparams['random_seed'],
                 batch_size=trainingparams['batch_size'],
                 hidden_activation=activation_functions[hyperparams['hidden_activation']],
                 use_cond_mask=hyperparams['use_cond_mask'],
                 direct_input_connect=hyperparams['direct_input_connect'],
                 direct_output_connect=hyperparams['direct_output_connect'],
                 update_rule=trainingparams['update_rule'],
                 dropout_rate=trainingparams['dropout_rate'],
                 weights_initialization=hyperparams['weights_initialization'],
                 mask_distribution=hyperparams['mask_distribution'])
    print(utils.get_done_text(start_time), "###")
    return model


def build_model_layer_pretraining(dataset, trainingparams, hyperparams, max_epochs):

    print('\n#### Pretraining layer {} ####'.format(1), end=' ')
    model = build_model(dataset, trainingparams, hyperparams, hyperparams['hidden_sizes'][:1])
    best_model, best_epoch, total_train_time = train_model(model, dataset, trainingparams['look_ahead'], trainingparams[
                                                           'shuffle_mask'], trainingparams['nb_shuffle_per_valid'], max_epochs, trainingparams['batch_size'], trainingparams['shuffling_type'])

    for i in range(2, len(hyperparams['hidden_sizes']) + 1):
        print('\n#### Pretraining layer {} ####'.format(i), end=' ')
        model = build_model(dataset, trainingparams, hyperparams, hyperparams['hidden_sizes'][:i])

        # Set pre-trained layers
        for j in range(i - 1):
            for paramIdx in range(len(best_model.layers[j].params)):
                model.layers[j].params[paramIdx].set_value(
                    best_model.layers[j].params[paramIdx].get_value())

        # Set pre-trained output
        for paramIdx in range(len(best_model.layers[-1].params)):
            if best_model.layers[-1].params[paramIdx] != best_model.layers[-1].W:
                model.layers[-
                             1].params[paramIdx].set_value(best_model.layers[-1].params[paramIdx].get_value())

        best_model, best_epoch, total_train_time = train_model(model, dataset, trainingparams['look_ahead'], trainingparams[
                                                               'shuffle_mask'], trainingparams['nb_shuffle_per_valid'], max_epochs, trainingparams['batch_size'], trainingparams['shuffling_type'])

    return best_model


def extract_embeddings(model,
                       dataset,
                       n_instances,
                       batch_size=100,
                       layer_ids=[0, 1, 2],
                       dtype=float):

    # embedding_sizes = [layer.n_out for layer in model.layers[:-1]]
    embedding_sizes = [model.layers[j].n_out for j in layer_ids]
    embedding_size = sum(embedding_sizes)
    feature_indexes = [0]
    for j, size in enumerate(embedding_sizes):
        feature_indexes += [size + feature_indexes[j]]

    print('\nTransforming data {} -> {} {}\n\t(layer embedding sizes {} )'.format(dataset.shape[1],
                                                                                  embedding_size,
                                                                                  feature_indexes,
                                                                                  embedding_sizes))
    repr_data = np.zeros((n_instances, embedding_size), dtype=dtype)
    repr_data[:] = np.nan

    nb_iterations = int(np.ceil(n_instances / batch_size))
    emb_data_start_t = perf_counter()
    for index in range(nb_iterations):
        emb_start_t = perf_counter()
        data_split = dataset[index * batch_size:(index + 1) * batch_size]
        for j in range(len(layer_ids)):
            # print('\t\tchecking j {} feature_indexes {}:{} layer_ids {}'.format(j,
            #                                                                     feature_indexes[
            #                                                                         j],
            #                                                                     feature_indexes[
            #                                                                         j + 1],
            #                                                                     layer_ids[j]))
            repr_data[index * batch_size:(index + 1) * batch_size,
                      feature_indexes[j]:feature_indexes[j + 1]] = model.embeddings(data_split,
                                                                                    layer_ids[j])
        emb_end_t = perf_counter()
        print('Processed batch {}/{} in {} secs'.format(index + 1,
                                                        nb_iterations,
                                                        emb_end_t - emb_start_t),
              end='                \r')
    emb_data_end_t = perf_counter()
    print('All dataset processed in {} secs'.format(emb_data_end_t - emb_data_start_t))

    assert not np.isnan(repr_data).any()
    return repr_data, feature_indexes


def parse_args(args):
    import argparse

    class GroupedAction(argparse.Action):

        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super(GroupedAction, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            group = self.container.title
            dest = self.dest
            groupspace = getattr(namespace, group, argparse.Namespace())
            setattr(groupspace, dest, values)
            setattr(namespace, group, groupspace)

    parser = argparse.ArgumentParser(description='Train the MADE model.')

    group_trainer = parser.add_argument_group('train')
    group_trainer.add_argument('dataset_name', action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument(
        'learning_rate', type=float, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument(
        'decrease_constant', type=float, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('max_epochs', type=lambda x: np.inf if x == "-1" else int(
        x), help="If -1 will run until convergence.", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument(
        'shuffle_mask', type=int, help="0=None, -1=No cycles.", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('shuffling_type', metavar='shuffling_type', choices=[
                               'Once', 'Full', 'Ordering', 'Connectivity'], help="Choosing how the masks will be shuffled: {%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('nb_shuffle_per_valid', type=int,
                               help="Only considered if shuffle_mask at -1.", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument(
        'batch_size', type=int, help="-1 will set to full batch.", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument(
        'look_ahead', type=int, help="Number of consecutive epochs without improvements before training stops.", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('pre_training', metavar='pre_training', type=eval, choices=[
                               False, True], help="{%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument(
        'pre_training_max_epoc', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('update_rule', metavar='update_rule', choices=[
                               'None', 'adadelta', 'adagrad', 'rmsprop', 'adam', 'adam_paper'], help="{%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument(
        'dropout_rate', type=float, help="%% of hidden neuron dropped with dropout.", action=GroupedAction, default=argparse.SUPPRESS)

    group_model = parser.add_argument_group('model')
    group_model.add_argument(
        'hidden_sizes', type=eval, help="ex: [500,200]", action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument(
        'random_seed', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('use_cond_mask', metavar='use_cond_mask', type=eval, choices=[
                             False, True], help="{%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('direct_input_connect', metavar='direct_input_connect', choices=[
                             "None", "Output", "Full"], help="{%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('direct_output_connect', metavar='direct_output_connect', type=eval, choices=[
                             False, True], help="{%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('hidden_activation', metavar='hidden_activation', choices=list(
        activation_functions.keys()), help="{%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('weights_initialization', metavar='weights_initialization', choices=[
                             x for x in WeightsInitializer.__dict__ if not x.startswith('_')], help="{%(choices)s}", action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('mask_distribution', type=float,
                             help="Gives some control over which input will have more connections. Ex: -1 will give more importance to the firsts inputs, 1 to the lasts and 0 uniform.", action=GroupedAction, default=argparse.SUPPRESS)

    parser.add_argument("--force", required=False, action='store_true',
                        help="Override instead of resuming training of pre-existing model with same arguments.")
    parser.add_argument("--name", required=False,
                        help="Set the name of the experiment instead of hashing it from the arguments.")
    parser.add_argument("--embeddings", required=False, type=str, default=None,
                        help="Extract embeddings from layers")
    parser.add_argument("--last-layer-embeddings", required=False, action='store_true',
                        help="Extract embeddings from last layer only")
    parser.add_argument("--save-model", required=False, type=str, default=None,
                        help="Dir to save the model as a pickle file")

    args = parser.parse_args()

    return args


def save_model_params(model, model_path):
    np.savez_compressed(
        os.path.join(model_path, "params"), model.parameters, model.update_rule.parameters)


def load_model_params(model, model_path):
    saved_parameters = np.load(os.path.join(model_path, "params.npz"))
    for i, param in enumerate(saved_parameters['arr_0']):
        model.parameters[i].set_value(param.get_value())

    for i, param in enumerate(saved_parameters['arr_1']):
        model.update_rule.parameters[i].set_value(param.get_value())

activation_functions = {
    "sigmoid": theano.tensor.nnet.sigmoid,
    # "hinge": lambda x: theano.tensor.maximum(x, 0.0),
    "hinge": theano.tensor.nnet.relu,
    "softplus": theano.tensor.nnet.softplus,
    "tanh": theano.tensor.tanh,
    "softsign": theano.tensor.nnet.nnet.softsign
}

if __name__ == '__main__':
    resume_mode = False

    #
    # Pars args from the shell
    args = parse_args(sys.argv)
    dataset_name = args.train.dataset_name
    hyperparams = vars(args.model)
    trainingparams = vars(args.train)

    #
    # Set the name of the experiment (remove the --force from the args to make
    # sure it will generate the same uid)
    if '--force' in sys.argv:
        sys.argv.remove('--force')
    experiment_name = args.name if args.name is not None else utils.generate_uid_from_string(
        ' '.join(sys.argv))

    #
    # Creating the experiments folder or resuming experiment
    save_path_experiment = os.path.join('./experiments/', experiment_name)
    if os.path.isdir(save_path_experiment):
        if not args.force:
            print("### Resuming experiment ({0}). ###\n".format(experiment_name))
            loaded_hyperparams = utils.load_dict_from_json_file(
                os.path.join(save_path_experiment, "hyperparams"))
            loaded_trainingparams = utils.load_dict_from_json_file(
                os.path.join(save_path_experiment, "trainingparams"))

            if loaded_trainingparams != trainingparams or loaded_hyperparams != hyperparams:
                print(
                    "The arguments provided are different than the one saved. Use --force if you are certain.\nQuitting.")
                exit()

            resume_mode = True

    else:
        os.makedirs(save_path_experiment)
        utils.save_dict_to_json_file(
            os.path.join(save_path_experiment, "hyperparams"), hyperparams)
        utils.save_dict_to_json_file(
            os.path.join(save_path_experiment, "trainingparams"), trainingparams)

    #
    # LOAD DATASET ####
    dataset = Dataset.get(dataset_name)
    if trainingparams['batch_size'] == -1:
        trainingparams['batch_size'] = dataset['train']['length']

    #
    # INITIALIZING LEARNER ####
    if trainingparams['pre_training']:
        model = build_model_layer_pretraining(
            dataset, trainingparams, hyperparams, trainingparams['pre_training_max_epoc'])
    else:
        model = build_model(dataset, trainingparams, hyperparams, hyperparams['hidden_sizes'])

    trainer_status = None

    # Not totally resumable if it was stopped during pre-training.
    if resume_mode:
        load_model_params(model, save_path_experiment)
        trainer_status = utils.load_dict_from_json_file(
            os.path.join(save_path_experiment, "trainer_status"))

    #
    # TRAINING LEARNER ####
    best_epoch, total_train_time = train_model(model, dataset, trainingparams['look_ahead'], trainingparams['shuffle_mask'], trainingparams[
                                               'nb_shuffle_per_valid'], trainingparams['max_epochs'], trainingparams['batch_size'], trainingparams['shuffling_type'], save_path_experiment, trainer_status)

    #
    # Loading best model
    load_model_params(model, save_path_experiment)

    #
    # EVALUATING BEST MODEL ####
    model_evaluation = {}
    print('\n### Evaluating best model from Epoch {0} ###'.format(best_epoch))
    for log_prob_func_name in ['test', 'valid', 'train']:
        if trainingparams['shuffle_mask'] > 0:
            model.reset(trainingparams['shuffling_type'])
        if log_prob_func_name == "train":
            model_evaluation[log_prob_func_name] = get_mean_error_and_std_final(model,
                                                                                model.train_log_prob_batch,
                                                                                dataset[log_prob_func_name][
                                                                                    'length'],
                                                                                trainingparams[
                                                                                    'shuffle_mask'],
                                                                                trainingparams[
                                                                                    'shuffling_type'],
                                                                                1000)
        else:
            model_evaluation[log_prob_func_name] = get_mean_error_and_std(model,
                                                                          model.__dict__['{}_log_prob'.format(
                                                                              log_prob_func_name)],
                                                                          dataset[log_prob_func_name][
                                                                              'length'],
                                                                          trainingparams[
                                                                              'shuffle_mask'],
                                                                          trainingparams[
                                                                              'shuffling_type'],
                                                                          1000)
        print("\tBest {1} error is : {0:.6f}".format(
            model_evaluation[log_prob_func_name][0], log_prob_func_name.upper()))

    #
    # WRITING RESULTS #####
    model_info = [trainingparams['learning_rate'], trainingparams['decrease_constant'], hyperparams['hidden_sizes'], hyperparams['random_seed'], hyperparams['hidden_activation'], trainingparams['max_epochs'], best_epoch, trainingparams['look_ahead'], trainingparams['batch_size'], trainingparams['shuffle_mask'], trainingparams['shuffling_type'], trainingparams['nb_shuffle_per_valid'], hyperparams['use_cond_mask'], hyperparams['direct_input_connect'], hyperparams[
        'direct_output_connect'], trainingparams['pre_training'], trainingparams['pre_training_max_epoc'], trainingparams['update_rule'], trainingparams['dropout_rate'], hyperparams['weights_initialization'], hyperparams['mask_distribution'], float(model_evaluation['train'][0]), float(model_evaluation['train'][1]), float(model_evaluation['valid'][0]), float(model_evaluation['valid'][1]), float(model_evaluation['test'][0]), float(model_evaluation['test'][1]), total_train_time]
    utils.write_result(dataset_name, model_info, experiment_name)

    if args.save_model:
        model_save_path = os.path.join(args.save_model, 'made.{}.{}.model'.format(dataset_name,
                                                                                  experiment_name))
        os.makedirs(args.save_model, exist_ok=True)
        model.to_pickle(model_save_path)

        # model2 = MADE.from_pickle(model_save_path + '.pklz')

    #
    # extracting embeddings
    if args.embeddings is not None:

        #
        # take them all?
        layer_ids = None
        if args.last_layer_embeddings:
            layer_ids = [len(model.layers[:-1]) - 1]
        else:
            layer_ids = [i for i in range(len(model.layers[:-1]))]

        print('Considering only layers {} for embeddings'.format(layer_ids))

        repr_save_path = os.path.join(args.embeddings,
                                      'made.{}.repr-data.pklz'.format(experiment_name))
        repr_data = []

        for dataset_split in ['train', 'valid', 'test']:

            split = dataset[dataset_split]['data'].get_value()
            print('\nProcessing split {} {}'.format(dataset_split,
                                                    dataset[dataset_split]['length']))
            repr_split, f = extract_embeddings(model,
                                               split,
                                               n_instances=dataset[dataset_split]['length'],
                                               batch_size=100,
                                               layer_ids=layer_ids,
                                               dtype=float)

            # repr_split2, f2 = extract_embeddings(model2,
            #                                      split,
            #                                      n_instances=dataset[dataset_split]['length'],
            #                                      batch_size=100,
            #                                      layer_ids=layer_ids,
            #                                      dtype=float)

            # assert_array_almost_equal(repr_split, repr_split2)

            repr_data.append(repr_split)
            #
            # saving it
        with gzip.open(repr_save_path, 'wb') as f:
            print('Saving splits to {}'.format(repr_save_path))
            pickle.dump(repr_data, f, protocol=4)

        # #
        # # getting last layer
        # lle = repr_data[0][:, -500:].astype(theano.config.floatX)
        # threshold = 0.5
        # p, inv_p, sp, inv_sp, pp, psp, ainv_p, ainv_sp, i_sp = model.predict(lle, threshold)

        # # p2, inv_p2, sp2, inv_sp2, pp2, psp2, ainv_p2, ainv_sp2, i_sp2 = model2.predict(lle,
        # #                                                                                threshold)

        # # assert_array_almost_equal(p, p2)
        # # assert_array_almost_equal(inv_p, inv_p2)
        # # assert_array_almost_equal(sp, sp2)
        # # assert_array_almost_equal(pp, pp2)
        # # assert_array_almost_equal(ainv_p, ainv_p2)
        # # assert_array_almost_equal(inv_sp, inv_sp2)
        # # assert_array_almost_equal(i_sp, i_sp2)

        # print(model.use(dataset['train']['data'].get_value()[:10], False)[:10])
        # print('data')
        # print(dataset['train']['data'].get_value()[:10].astype(int))
        # print('p')
        # print(p[:10])
        # print('inv p')
        # print(inv_p[:10])
        # print('ainv p')
        # print(ainv_p[:10])
        # print('sp')
        # print(sp[:10])
        # print('inv sp')
        # print(inv_sp[:10])
        # print('ainv sp')
        # print(ainv_sp[:10])
        # print('pp')
        # print(pp[:10])
        # print('psp')
        # print(psp[:10])
        # print('iter sp')
        # print(i_sp[:10])
