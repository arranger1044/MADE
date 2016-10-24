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


def majority_ensemble(preds_tensor):

    ens_preds = np.zeros((preds_tensor.shape[0],
                          preds_tensor.shape[1]), dtype=preds_tensor.dtype)

    for i in range(preds_tensor.shape[0]):
        for j in range(preds_tensor.shape[1]):
            p_j_ens = preds_tensor[i, j]
            values, counts = np.unique(p_j_ens, return_counts=True)
            mode_ind = np.argmax(counts)
            ens_preds[i, j] = values[mode_ind]

    return ens_preds


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
        # log_prob = T.sum(target * T.log(output) + (1 - target) * T.log(1 - output), axis=1)
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

        #
        # NOTE: the predict method (for decoding) is possible only when there is no skip
        # connections to the output layer
        if direct_input_connect == 'None' and not direct_output_connect:
            print('No skip connections! defining decoding function')
            pred_threshold = T.vector()
            last_layer_embeddings = T.matrix(name="ll-embeddings")
            output_probs = T.matrix(name="output-probs")
            # T.dot(last_layer_embeddings, self.layers[-1].W) + self.layers[-1].b
            pred_probs = output
            predictions = T.switch(pred_probs < pred_threshold, 0, 1)
            thresholded_output = T.switch(output_probs < pred_threshold, 0, 1)

            self.predict_probs = theano.function(name='predict_probs',
                                                 inputs=[last_layer_embeddings, is_train],
                                                 outputs=pred_probs,
                                                 givens={self.layers[-1].input:
                                                         last_layer_embeddings},
                                                 on_unused_input='ignore')
            self.threshold_probs = theano.function(name='threshold_probs',
                                                   inputs=[output_probs, pred_threshold],
                                                   outputs=thresholded_output,
                                                   on_unused_input='ignore')
            self.predict_func = theano.function(name='predict',
                                                inputs=[last_layer_embeddings, pred_threshold],
                                                outputs=predictions,
                                                givens={self.layers[-1].input:
                                                        last_layer_embeddings},
                                                on_unused_input='ignore')
        else:
            self.predict_func = None
            print('Skip connections detected! decoding will fail!')

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

    def embeddings_ensemble(self, input, layer_id, shuffle_mask, shuffling_type, nb_shuffle=1):
        """
        Return the embedding representation of input as the activations
        of the neurons in layer specified by layer_id
        one for each shuffling
        """

        if shuffle_mask > 0:
            self.reset(shuffling_type)

        if shuffle_mask > 0:
            nb_shuffle = shuffle_mask + 1

        if not shuffle_mask:
            nb_shuffle = 1

        embeddings = []
        for i in range(nb_shuffle):
            if shuffle_mask:
                self.shuffle(shuffling_type)

            emb = self.embedding_funcs[layer_id](input, False)
            embeddings.append(emb)

        return np.dstack(embeddings)

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

    def estimate_threshold_(self, probs, data, feature_wise=False):
        """
        Estimating an array of thresholds (label_wise or globally) for
        setting predictions to 0 or 1, based on some train data
        """
        thresholds = None
        n_features = data.shape[1]
        n_samples = data.shape[0]

        n_points = n_samples * n_features

        print('p', n_points, n_samples, n_features)
        if not feature_wise:
            #
            # flatten and sort probs
            sorted_probs_array = np.sort(probs, axis=None)
            #
            # compute proportion of zeros in data
            n_zeros = min(int(n_points - data.sum()), n_points - 1)
            #
            #
            threshold = sorted_probs_array[n_zeros]
            thresholds = np.array([threshold for i in range(n_features)])
        else:
            #
            # sort column wise
            sorted_probs_array = np.sort(probs, axis=0)
            #
            # compute proportion of zeros in data
            n_zeros = np.minimum((n_samples - data.sum(axis=0)).astype(int),
                                 n_samples - 1)
            print(n_zeros.shape, n_zeros, data.sum(axis=0))
            thresholds = sorted_probs_array[n_zeros, np.arange(n_features)]

        assert thresholds.ndim == 1 and thresholds.shape[0] == n_features
        print('Estimated thresholds {}'.format(thresholds))
        return thresholds

    def predict(self, embeddings, threshold=None, data=None, feature_wise=False):
        """
        Returning the predictions of the decoder (\hat(x))
        given the last hidden layer embeddings

        Either a threshold must be given,
        or a training set on which to compute the threshold
        """

        assert threshold is not None or data is not None
        #
        # taking only the last layer embeddings, assuming them to be ordered to be the last
        n_last_layer_neurons = self.layers[-2].n_out
        ll_embeddings = embeddings[:, -n_last_layer_neurons:].astype(np.float32)
        print('Taking only last {} activations ({})'.format(n_last_layer_neurons,
                                                            ll_embeddings.shape))

        #
        # predicting the output probabilities give the last hidden layer embeddings
        print(ll_embeddings.shape)
        pred_probs = self.predict_probs(ll_embeddings, False)

        #
        # computing the threshold?
        if threshold is None:
            threshold = self.estimate_threshold_(pred_probs, data, feature_wise=feature_wise)

        #
        # make predictions and return the threshold as well
        return threshold, self.threshold_probs(pred_probs, threshold)

    def predict_ensemble(self, embeddings, threshold, data, feature_wise,
                         shuffle_mask, shuffling_type, nb_shuffle=1):
        if shuffle_mask > 0:
            self.reset(shuffling_type)

        if shuffle_mask > 0:
            nb_shuffle = shuffle_mask + 1

        if not shuffle_mask:
            nb_shuffle = 1

        predictions = []
        for i in range(nb_shuffle):
            if shuffle_mask:
                self.shuffle(shuffling_type)

            emb = embeddings[:, :, i]

            th, preds = self.predict(emb, threshold, data, feature_wise)
            predictions.append(preds)

        preds_tensor = np.dstack(predictions)

        #
        # taking the most voted prediction
        return threshold, majority_ensemble(preds_tensor)
