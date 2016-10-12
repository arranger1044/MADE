from __future__ import division
from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T
# import dill as pickle
import pickle

import gzip

from .update_rules import DecreasingLearningRate, AdaGrad, AdaDelta, RMSProp, Adam, Adam_paper
# , MaskGenerator
from .layer_types import ConditionningMaskedLayer, dropoutLayerDecorator, DirectInputConnectConditionningMaskedLayer, DirectOutputInputConnectConditionningMaskedOutputLayer
from .mask_generator import MaskGenerator
from .weights_initializer import WeightsInitializer


class SeedGenerator(object):
    # This subclass purpose is to maximize randomness and still keep reproducibility

    def __init__(self, random_seed):
        self.rng = np.random.mtrand.RandomState(random_seed)

    def get(self):
        return self.rng.randint(42424242)


class MADE(object):

    def __init__(self, dataset,
                 learning_rate=0.001,
                 decrease_constant=0,
                 hidden_sizes=[500],
                 random_seed=1234,
                 batch_size=1,
                 hidden_activation=T.nnet.sigmoid,
                 use_cond_mask=False,
                 direct_input_connect="None",
                 direct_output_connect=False,
                 update_rule="None",
                 dropout_rate=0,
                 weights_initialization="Uniform",
                 mask_distribution=0):

        input_size = dataset['input_size']
        self.shuffled_once = False

        self.seed_generator = SeedGenerator(random_seed)

        self.trng = T.shared_randomstreams.RandomStreams(self.seed_generator.get())

        # Get the weights initializer by string name
        weights_initialization = getattr(
            WeightsInitializer(self.seed_generator.get()), weights_initialization)

        # Building the model's graph
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        is_train = T.bscalar(name="is_train")

        # Initialize the mask
        self.mask_generator = MaskGenerator(
            input_size, hidden_sizes, mask_distribution, self.seed_generator.get())

        # Initialize layers
        input_layer = ConditionningMaskedLayer(layerIdx=0,
                                               input=input,
                                               n_in=input_size,
                                               n_out=hidden_sizes[0],
                                               activation=hidden_activation,
                                               weights_initialization=weights_initialization,
                                               mask_generator=self.mask_generator,
                                               use_cond_mask=use_cond_mask)
        self.layers = [dropoutLayerDecorator(input_layer, self.trng, is_train, dropout_rate)]
        # Now the hidden layers
        for i in range(1, len(hidden_sizes)):
            previous_layer = self.layers[i - 1]
            hidden_layer = DirectInputConnectConditionningMaskedLayer(layerIdx=i,
                                                                      input=previous_layer.output,
                                                                      n_in=hidden_sizes[i - 1],
                                                                      n_out=hidden_sizes[i],
                                                                      activation=hidden_activation,
                                                                      weights_initialization=weights_initialization,
                                                                      mask_generator=self.mask_generator,
                                                                      use_cond_mask=use_cond_mask,
                                                                      direct_input=input if direct_input_connect == "Full" and previous_layer.output != input else None)
            self.layers += [dropoutLayerDecorator(hidden_layer, self.trng, is_train, dropout_rate)]
        # And the output layer
        outputLayerIdx = len(self.layers)
        previous_layer = self.layers[outputLayerIdx - 1]
        self.layers += [DirectOutputInputConnectConditionningMaskedOutputLayer(layerIdx=outputLayerIdx,
                                                                               input=previous_layer.output,
                                                                               n_in=hidden_sizes[
                                                                                   outputLayerIdx - 1],
                                                                               n_out=input_size,
                                                                               activation=T.nnet.sigmoid,
                                                                               weights_initialization=weights_initialization,
                                                                               mask_generator=self.mask_generator,
                                                                               use_cond_mask=use_cond_mask,
                                                                               direct_input=input if (
                                                                                   direct_input_connect == "Full" or direct_input_connect == "Output") and previous_layer.output != input else None,
                                                                               direct_outputs=[(layer.layer_idx, layer.n_in, layer.input) for layerIdx, layer in enumerate(self.layers[1:-1])] if direct_output_connect else [])]

        # The loss function
        output = self.layers[-1].output
        pre_output = self.layers[-1].lin_output
        log_prob = - \
            T.sum(T.nnet.softplus(-target * pre_output + (1 - target) * pre_output), axis=1)
        loss = (-log_prob).mean()

        # How to update the parameters
        self.parameters = [param for layer in self.layers for param in layer.params]
        parameters_gradient = T.grad(loss, self.parameters)

        # Initialize update_rule
        if update_rule == "None":
            self.update_rule = DecreasingLearningRate(learning_rate, decrease_constant)
        elif update_rule == "adadelta":
            self.update_rule = AdaDelta(decay=decrease_constant, epsilon=learning_rate)
        elif update_rule == "adagrad":
            self.update_rule = AdaGrad(learning_rate=learning_rate)
        elif update_rule == "rmsprop":
            self.update_rule = RMSProp(learning_rate=learning_rate, decay=decrease_constant)
        elif update_rule == "adam":
            self.update_rule = Adam(learning_rate=learning_rate)
        elif update_rule == "adam_paper":
            self.update_rule = Adam_paper(learning_rate=learning_rate)
        updates = self.update_rule.get_updates(list(zip(self.parameters, parameters_gradient)))

        # How to to shuffle weights
        masks_updates = [
            layer_mask_update for layer in self.layers for layer_mask_update in layer.shuffle_update]
        self.update_masks = theano.function(name='update_masks',
                                            inputs=[],
                                            updates=masks_updates)
        #
        # Functions to train and use the model
        index = T.lscalar()
        self.learn = theano.function(name='learn',
                                     inputs=[index, is_train],
                                     outputs=loss,
                                     updates=updates,
                                     givens={input: dataset['train']['data'][
                                         index * batch_size:(index + 1) * batch_size], target: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]},
                                     on_unused_input='ignore')  # ignore for when dropout is absent

        self.use = theano.function(name='use',
                                   inputs=[input, is_train],
                                   outputs=output,
                                   on_unused_input='ignore')  # ignore for when dropout is absent

        # Test functions
        self.valid_log_prob = theano.function(name='valid_log_prob',
                                              inputs=[is_train],
                                              outputs=log_prob,
                                              givens={
                                                  input: dataset['valid']['data'], target: dataset['valid']['data']},
                                              on_unused_input='ignore')  # ignore for when dropout is absent
        self.train_log_prob = theano.function(name='train_log_prob',
                                              inputs=[is_train],
                                              outputs=log_prob,
                                              givens={
                                                  input: dataset['train']['data'], target: dataset['train']['data']},
                                              on_unused_input='ignore')  # ignore for when dropout is absent
        self.train_log_prob_batch = theano.function(name='train_log_prob_batch',
                                                    inputs=[index, is_train],
                                                    outputs=log_prob,
                                                    givens={input: dataset['train']['data'][
                                                        index * 1000:(index + 1) * 1000], target: dataset['train']['data'][index * 1000:(index + 1) * 1000]},
                                                    on_unused_input='ignore')  # ignore for when dropout is absent
        self.test_log_prob = theano.function(name='test_log_prob',
                                             inputs=[is_train],
                                             outputs=log_prob,
                                             givens={
                                                 input: dataset['test']['data'], target: dataset['test']['data']},
                                             on_unused_input='ignore')  # ignore for when dropout is absent

        # Functions for verify gradient
        self.useloss = theano.function(name='useloss',
                                       inputs=[input, target, is_train],
                                       outputs=loss,
                                       on_unused_input='ignore')  # ignore for when dropout is absent
        self.learngrad = theano.function(name='learn',
                                         inputs=[index, is_train],
                                         outputs=parameters_gradient,
                                         givens={input: dataset['train']['data'][
                                             index * batch_size:(index + 1) * batch_size], target: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]},
                                         on_unused_input='ignore')  # ignore for when dropout is absent

        #
        # adding functions to extract embeddings from each layer
        self.embedding_funcs = [theano.function(name='embedding-{}'.format(i),
                                                inputs=[input, is_train],
                                                outputs=layer.output,
                                                # givens={input: dataset['train']['data'][
                                                # index * batch_size:(index + 1) * batch_size]},
                                                on_unused_input='ignore')
                                for i, layer in enumerate(self.layers[:-1])]

        pred_threshold = T.scalar()
        last_layer_embeddings = T.matrix(name="ll-embeddings")
        pred_probs = output  # T.dot(last_layer_embeddings, self.layers[-1].W) + self.layers[-1].b
        predictions = T.switch(pred_probs < pred_threshold, 0, 1)

        self.predict_func = theano.function(name='predict',
                                            inputs=[last_layer_embeddings, pred_threshold],
                                            outputs=[output, predictions],
                                            givens={self.layers[-1].input: last_layer_embeddings},
                                            on_unused_input='ignore')

    def shuffle(self, shuffling_type):
        if shuffling_type == "Once" and self.shuffled_once is False:
            self.mask_generator.shuffle_ordering()
            self.mask_generator.sample_connectivity()
            self.update_masks()
            self.shuffled_once = True
            return

        if shuffling_type in ["Ordering", "Full"]:
            self.mask_generator.shuffle_ordering()
        if shuffling_type in ["Connectivity", "Full"]:
            self.mask_generator.sample_connectivity()
        self.update_masks()

    def reset(self, shuffling_type, last_shuffle=0):
        self.mask_generator.reset()

        # Always do a first shuffle so that the natural order does not gives us an edge
        self.shuffle("Full")

        # Set the mask to the requested shuffle
        for i in range(last_shuffle):
            self.shuffle(shuffling_type)

    def sample(self, nb_samples=1, mask_id=0):
        rng = np.random.mtrand.RandomState(self.seed_generator.get())

        self.reset("Full", mask_id)

        swap_order = self.mask_generator.ordering.get_value()
        input_size = self.layers[0].W.shape[0].eval()

        samples = np.zeros((nb_samples, input_size), theano.config.floatX)

        for i in range(input_size):
            inv_swap = np.where(swap_order == i)[0][0]
            out = self.use(samples, False)
            rng.binomial(p=out[:, inv_swap], n=1)
            samples[:, inv_swap] = rng.binomial(p=out[:, inv_swap], n=1)

        return samples

    def embeddings(self, input, layer_id):
        """
        Return the embedding representation of input as the activations
        of the neurons in layer specified by layer_id
        """
        emb = self.embedding_funcs[layer_id](input, False)
        return emb

    def to_pickle(self, file_path, compress=True):
        """
        Exporting the theano model to a (possibly compressed) pickle file
        """

        model_file = None
        if compress:
            file_path = file_path if file_path.endswith('.pklz') else '{}.pklz'.format(file_path)
            model_file = gzip.open(file_path, 'wb')
        else:
            file_path = file_path if file_path.endswith('.pkl') else '{}.pkl'.format(file_path)
            model_file = open(file_path, 'wb')

        print('Saving MADE model to pickle file: {}'.format(file_path))

        pickle.dump(self, model_file, protocol=4)
        model_file.close()

    @classmethod
    def from_pickle(cls, file_path, compressed=True):
        model_file = None
        if compressed:
            model_file = gzip.open(file_path, 'rb')
        else:
            model_file = open(file_path, 'rb')

        return pickle.load(model_file)

    # def predict(self, last_layer_embeddings, threshold):
    #     """
    #     Returning the predictions of the decoder (\hat(x))
    #     given the last hidden layer embeddings
    #     """

    #     swap_order = self.mask_generator.ordering.get_value()

    #     input_size = self.layers[0].W.shape[0].eval()

    #     output_probs, preds = self.predict_func(last_layer_embeddings, threshold)

    #     embs = np.zeros(last_layer_embeddings.shape, dtype=last_layer_embeddings.dtype)

    #     print('output: {}'.format(output_probs))
    #     print('preds: {}'.format(preds))

    #     rng = np.random.mtrand.RandomState(self.seed_generator.get())

    #     inv_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     iter_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     sampled_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     p_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     p_sampled_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     inv_sampled_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     inv_output_probs = np.zeros(output_probs.shape, dtype=output_probs.dtype)
    #     ainv_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     ainv_sampled_preds = np.zeros(preds.shape, dtype=preds.dtype)
    #     ainv_output_probs = np.zeros(output_probs.shape, dtype=output_probs.dtype)
    #     print(swap_order)

    #     # for i in range(input_size):
    #     #     inv_swap = np.where(swap_order == i)[0][0]
    #     #     out = self.use(samples, False)
    #     #     rng.binomial(p=out[:, inv_swap], n=1)
    #     #     samples[:, inv_swap] = rng.binomial(p=out[:, inv_swap], n=1)

    #     for i in range(input_size):
    #         inv_swap = np.where(swap_order == i)[0][0]
    #         o, p = self.predict_func(embs, threshold)
    #         embs[:, inv_swap] = last_layer_embeddings[:, inv_swap]
    #         iter_preds[:, inv_swap] = rng.binomial(p=o[:, inv_swap], n=1)

    #         print('i {} inv {}'.format(i, inv_swap))
    #         inv_preds[:, i] = preds[:, inv_swap]
    #         inv_output_probs[:, i] = output_probs[:, inv_swap]
    #         inv_sampled_preds[:, i] = rng.binomial(p=output_probs[:, inv_swap], n=1)
    #         ainv_preds[:, inv_swap] = preds[:, i]
    #         ainv_output_probs[:, inv_swap] = output_probs[:, i]
    #         ainv_sampled_preds[:, inv_swap] = rng.binomial(p=output_probs[:, i], n=1)
    #         sampled_preds[:, i] = rng.binomial(p=output_probs[:, i], n=1)
    #         p_sampled_preds[:, inv_swap] = rng.binomial(p=output_probs[:, inv_swap], n=1)
    #         p_preds[:, inv_swap] = preds[:, inv_swap]
    #     print('inv output: {}'.format(inv_output_probs))
    #     print('inv preds: {}'.format(inv_preds))
    #     print('inv sampled preds: {}'.format(inv_sampled_preds))
    #     print(inv_preds[:10])

    # return preds, inv_preds, sampled_preds, inv_sampled_preds, p_preds,
    # p_sampled_preds, ainv_preds, ainv_sampled_preds, iter_preds
