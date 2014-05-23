import cPickle
import os
import sys
import time
import socket
import random

import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared

from logistic_regression import LogisticRegression 
from dataset_sentences_iterator import DatasetSentencesIteratorPhnSpkr
from mlp import HiddenLayer
from relu_rbm import RBM
from relu_grbm import GRBM
from prep_timit import load_data

#DATASET = '/home/gsynnaeve/datasets/TIMIT'
#DATASET = '/media/bigdata/TIMIT'
#DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/wo_sa'
DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split'
if socket.gethostname() == "syhws-MacBook-Pro.local":
    DATASET = '/Users/gabrielsynnaeve/postdoc/datasets/TIMIT_train_dev_test'
N_FEATURES = 40  # filterbanks
N_FRAMES = 13  # HAS TO BE AN ODD NUMBER 
               #(same number before and after center frame)
PRELEARNING_RATE_DENOMINATOR_FOR_GAUSSIAN = 50. # we take a lower learning rate
                                             # for the Gaussian RBM
BORROW = True
output_file_name = 'dbn_spk_phn_13_frames'



class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=N_FEATURES * N_FRAMES,
                 hidden_layers_sizes=[1024, 1024], n_phn=62 * 3, n_spkr=1,
                 rho=0.90, eps=1.E-6):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        #self._rho = shared(numpy.cast['float32'](rho), name='rho')  # for adadelta
        #self._eps = shared(numpy.cast['float32'](eps), name='eps')  # for adadelta
        self._rho = rho
        self._eps = eps
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.fmatrix('x')  # the data is presented as rasterized images
        self.y_phn = T.ivector('y_phn')  # the labels are presented as 1D vector
                                 # of [int] labels
        self.y_spkr = T.ivector('y_spkr')  # the labels are presented as 1D vector
                                 # of [int] labels

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)
            self._accugrads.extend([shared(value=numpy.zeros((input_size, hidden_layers_sizes[i]), dtype='float32'), name='accugrad_W', borrow=True), shared(value=numpy.zeros((hidden_layers_sizes[i], ), dtype='float32'), name='accugrad_b', borrow=True)]) # TODO
            self._accudeltas.extend([shared(value=numpy.zeros((input_size, hidden_layers_sizes[i]), dtype='float32'), name='accudelta_W', borrow=True), shared(value=numpy.zeros((hidden_layers_sizes[i], ), dtype='float32'), name='accudelta_b', borrow=True)]) # TODO

            # Construct an RBM that shared weights with this layer
            if i == 0:
                rbm_layer = GRBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayerPhn = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_phn)
        self.params.extend(self.logLayerPhn.params)
        self._accugrads.extend([shared(value=numpy.zeros((hidden_layers_sizes[-1], n_phn), dtype='float32'), name='accugrad_W', borrow=True), shared(value=numpy.zeros((n_phn, ), dtype='float32'), name='accugrad_b', borrow=True)]) # TODO
        self._accudeltas.extend([shared(value=numpy.zeros((hidden_layers_sizes[-1], n_phn), dtype='float32'), name='accudelta_W', borrow=True), shared(value=numpy.zeros((n_phn, ), dtype='float32'), name='accudelta_b', borrow=True)]) # TODO
        self.logLayerSpkr = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_spkr)
        self.params.extend(self.logLayerSpkr.params)
        self._accugrads.extend([shared(value=numpy.zeros((hidden_layers_sizes[-1], n_spkr), dtype='float32'), name='accugrad_W', borrow=True), shared(value=numpy.zeros((n_spkr, ), dtype='float32'), name='accugrad_b', borrow=True)]) # TODO
        self._accudeltas.extend([shared(value=numpy.zeros((hidden_layers_sizes[-1], n_spkr), dtype='float32'), name='accudelta_W', borrow=True), shared(value=numpy.zeros((n_spkr, ), dtype='float32'), name='accudelta_b', borrow=True)]) # TODO

        self.finetune_cost_sum_phn = self.logLayerPhn.negative_log_likelihood_sum(self.y_phn)
        self.finetune_cost_sum_spkr = self.logLayerSpkr.negative_log_likelihood_sum(self.y_spkr)
        self.finetune_cost_phn = self.logLayerPhn.negative_log_likelihood(self.y_phn)
        self.finetune_cost_spkr = self.logLayerSpkr.negative_log_likelihood(self.y_spkr)

        self.errors_phn = self.logLayerPhn.errors(self.y_phn)
        self.errors_spkr = self.logLayerSpkr.errors(self.y_spkr)

    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        cost = self.finetune_cost_sum
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, self.params)

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate 

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y_phn = T.ivector('batch_y_phn')
        batch_y_spkr = T.ivector('batch_y_spkr')
        cost_phn = self.finetune_cost_sum_phn
        cost_spkr = self.finetune_cost_sum_spkr
        # compute the gradients with respect to the model parameters
        gparams_phn = T.grad(cost_phn, self.params[:-2])
        gparams_spkr = T.grad(cost_spkr, self.params[:-4] + self.params[-2:])

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads[:-2],
                self._accudeltas[:-2], self.params[:-2], gparams_phn):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad
        for accugrad, accudelta, param, gparam in zip(self._accugrads[:-4] + self._accugrads[-2:], self._accudeltas[:-4] + self._accudeltas[-2:], self.params[:-4] + self.params[-2:], gparams_spkr):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y_phn),
            theano.Param(batch_y_spkr)],
            outputs=(cost_phn, cost_spkr),
            updates=updates,
            givens={self.x: batch_x, self.y_phn: batch_y_phn, self.y_spkr: batch_y_spkr})

        return train_fn

    def get_adadelta_trainers(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y_phn = T.ivector('batch_y_phn')
        batch_y_spkr = T.ivector('batch_y_spkr')
        #cost_phn = self.finetune_cost_sum_phn
        cost_phn = self.finetune_cost_phn
        #cost_spkr = self.finetune_cost_sum_spkr
        cost_spkr = self.finetune_cost_spkr
        # compute the gradients with respect to the model parameters
        gparams_phn = T.grad(cost_phn, self.params[:-2])
        gparams_spkr = T.grad(cost_spkr, self.params[:-4] + self.params[-2:])

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads[:-2],
                self._accudeltas[:-2], self.params[:-2], gparams_phn):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad
        train_fn_phn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y_phn)],
            outputs=cost_phn,
            updates=updates,
            givens={self.x: batch_x, self.y_phn: batch_y_phn})

        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads[:-4] + self._accugrads[-2:], self._accudeltas[:-4] + self._accudeltas[-2:], self.params[:-4] + self.params[-2:], gparams_spkr):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad
        train_fn_spkr = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y_spkr)],
            outputs=cost_spkr,
            updates=updates,
            #givens={self.x: batch_x[20:24,:], self.y_spkr: batch_y_spkr[20:24]})
            givens={self.x: batch_x, self.y_spkr: batch_y_spkr})

        return train_fn_phn, train_fn_spkr

    def train_only_classif(self):
        batch_x = T.fmatrix('batch_x')
        batch_y_phn = T.ivector('batch_y_phn')
        batch_y_spkr = T.ivector('batch_y_spkr')
        #cost_phn = self.finetune_cost_sum_phn
        cost_phn = self.finetune_cost_phn
        #cost_spkr = self.finetune_cost_sum_spkr
        cost_spkr = self.finetune_cost_spkr
        # compute the gradients with respect to the model parameters
        gparams_phn = T.grad(cost_phn, self.params[-4:-2])
        gparams_spkr = T.grad(cost_spkr, self.params[-2:])

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads[-4:-2],
                self._accudeltas[-4:-2], self.params[-4:-2], gparams_phn):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad
        train_fn_phn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y_phn)],
            outputs=cost_phn,
            updates=updates,
            givens={self.x: batch_x, self.y_phn: batch_y_phn})

        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads[-2:], self._accudeltas[-2:], self.params[-2:], gparams_spkr):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad
        train_fn_spkr = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y_spkr)],
            outputs=cost_spkr,
            updates=updates,
            #givens={self.x: batch_x[20:24,:], self.y_spkr: batch_y_spkr[20:24]})
            givens={self.x: batch_x, self.y_spkr: batch_y_spkr})

        return train_fn_phn, train_fn_spkr

    def score_classif(self, given_set):
        """ Returns functions to get current classification scores. """
        batch_x = T.fmatrix('batch_x')
        batch_y_phn = T.ivector('batch_y_phn')
        batch_y_spkr = T.ivector('batch_y_spkr')
        score = theano.function(inputs=[theano.Param(batch_x), theano.Param(batch_y_phn), theano.Param(batch_y_spkr)],
                outputs=(self.errors_phn, self.errors_spkr),
                givens={self.x: batch_x, self.y_phn: batch_y_phn, self.y_spkr: batch_y_spkr})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(batch_x, batch_y_phn, batch_y_spkr) for batch_x, batch_y_phn, batch_y_spkr in given_set]

        return scoref


def test_DBN(finetune_lr=0.01, pretraining_epochs=0,
             pretrain_lr=0.01, k=1, training_epochs=500, # TODO 100+
             dataset=DATASET, batch_size=100):
    """

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    print "loading dataset from", dataset
    datasets = load_data(dataset, nframes=1, features='fbank', scaling='student', cv_frac='fixed', speakers=True, numpy_array_only=True) 

    train_set_x, train_set_y = datasets[0]  # if speakers, do test/test/test
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print "dataset loaded!"
    print "train set size", train_set_x.shape[0]
    print "validation set size", valid_set_x.shape[0]
    print "test set size", test_set_x.shape[0]
    print "#spkrs in train", len(set(train_set_y[1]))
    print "#spkrs in valid", len(set(valid_set_y[1]))
    print "#spkrs in test", len(set(test_set_y[1]))
#    test_set_y = test_set_y[0], test_set_y[1] - test_set_y[1].min()
#    print "checking that both y_phn and y_spkr are 0-indexed (for the Softmax)"
#    print "y_phn min:", test_set_y[0].min(), 
#    print "y_spkr min:", test_set_y[1].min() 
#    assert test_set_y[0].min() == 0
#    assert test_set_y[1].min() == 0

    to_int = {}
    with open('timit_to_int_and_to_state_dicts_tuple.pickle') as f:  # TODO
        to_int, _ = cPickle.load(f)
    train_set_iterator = DatasetSentencesIteratorPhnSpkr(train_set_x,
            train_set_y, to_int, N_FRAMES)
    valid_set_iterator = DatasetSentencesIteratorPhnSpkr(valid_set_x,
            valid_set_y, to_int, N_FRAMES)
    test_set_iterator = DatasetSentencesIteratorPhnSpkr(test_set_x,
            test_set_y, to_int, N_FRAMES)

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=N_FRAMES * N_FEATURES,
              hidden_layers_sizes=[2048, 2048, 2048],
#              n_phn=len(set(test_set_y[0])), n_spkr=len(set(test_set_y[1])))
              n_phn=len(set(test_set_y[0])),
              n_spkr=len(set(train_set_y[1])) + len(set(valid_set_y[1])) + len(set(test_set_y[1])))

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    #train_fn = dbn.get_adadelta_trainer()
    train_fn_phn, train_fn_spkr = dbn.get_adadelta_trainers()
    train_clf_phn, train_clf_spkr = dbn.train_only_classif()
    train_scoref = dbn.score_classif(train_set_iterator)
    valid_scoref = dbn.score_classif(valid_set_iterator)
    test_scoref = dbn.score_classif(test_set_iterator)

    print '... finetuning the model'
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless TODO
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant

    best_validation_loss_phn = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        avg_costs_phn = []
        avg_costs_spkr = []
        for iteration, (x, y_phn, y_spkr) in enumerate(train_set_iterator):
            if random.random() > 0.8:  # TODO play with this ratio
                avg_cost_phn = train_fn_phn(x, y_phn)
                avg_costs_phn.append(avg_cost_phn)
            else:
                avg_cost_spkr = train_fn_spkr(x, y_spkr)
                avg_costs_spkr.append(avg_cost_spkr)

            #avg_cost_phn = train_clf_phn(x, y_phn)
            #avg_costs_phn.append(avg_cost_phn)

            #avg_cost_spkr = train_clf_spkr(x, y_spkr)
            #avg_costs_spkr.append(avg_cost_spkr)


            #avg_cost = train_fn(x, y_phn, y_spkr)
            #avg_costs_phn.append(avg_cost[0])
            #avg_costs_spkr.append(avg_cost[1])
        print('  epoch %i, avg costs phn %f' % \
              (epoch, numpy.mean(avg_costs_phn)))
        print('  epoch %i, avg costs spkr %f' % \
              (epoch, numpy.mean(avg_costs_spkr)))
        zipped = zip(*train_scoref())
        print('  epoch %i, training error phn %f %%' % \
              (epoch, numpy.mean(zipped[0]) * 100.))
        print('  epoch %i, training error spkr %f %%' % \
              (epoch, numpy.mean(zipped[1]) * 100.))

        # we check the validation loss on every epoch
        validation_losses = valid_scoref()
        this_phn_validation_loss = numpy.mean(zip(*validation_losses)[0])  # TODO this is a mean of means (with different lengths)
        print('  epoch %i, validation error phn %f %%' % \
              (epoch, this_phn_validation_loss * 100.))
        # if we got the best validation score until now
        if this_phn_validation_loss < best_validation_loss_phn:
            with open(output_file_name + '.pickle', 'w') as f:
                cPickle.dump(dbn, f)
            # improve patience if loss improvement is good enough
            if (this_phn_validation_loss < best_validation_loss_phn *
                improvement_threshold):
                patience = max(patience, iteration * patience_increase)
            # save best validation score and iteration number
            best_validation_loss = this_phn_validation_loss
            # test it on the test set
            test_losses = test_scoref()
            test_score_phn = numpy.mean(zip(*test_losses)[0])  # TODO this is a mean of means (with different lengths)
            print(('  epoch %i, test error phn of '
                   'best model %f %%') %
                  (epoch, test_score_phn * 100.))
        if patience <= iteration:  # TODO correct that
            done_looping = True
            break

    end_time = time.clock()
    print(('Optimization complete with best validation score phn of %f %%, '
           'with test performance phn %f %%') %
                 (best_validation_loss_phn * 100., test_score_phn * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    with open(output_file_name + '.pickle', 'w') as f:
        cPickle.dump(dbn, f)


if __name__ == '__main__':
    test_DBN()
