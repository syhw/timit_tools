import cPickle
import os
import sys
import time

import numpy
from collections import OrderedDict, defaultdict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared

from logistic_timit import LogisticRegression 
from mlp import HiddenLayer
from relu_rbm import RBM
from relu_grbm import GRBM
from prep_timit import load_data

#DATASET = '/home/gsynnaeve/datasets/TIMIT'
#DATASET = '/media/bigdata/TIMIT'
#DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/wo_sa'
DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split'
#DATASET = '/Users/gabrielsynnaeve/postdoc/datasets/TIMIT_train_dev_test'
N_FEATURES = 40  # filterbanks
N_FRAMES = 13  # HAS TO BE AN ODD NUMBER 
               #(same number before and after center frame)
PRELEARNING_RATE_DENOMINATOR_FOR_GAUSSIAN = 50. # we take a lower learning rate
                                             # for the Gaussian RBM
MIN_FRAMES_PER_SENTENCE = 10
BORROW = True
output_file_name = 'dbn_ReLu_2496_units_13_frames'


class DatasetSentencesIterator(object):
    """ An iterator of setences of the dataset. """

    def __init__(self, x, y, phn_to_st, nframes=1):
        self._x = x
        self._y = y
        self._start_end = [[0]]
        self._nframes = nframes
        self._memoized_x = defaultdict(lambda: {})
        i = 0
        for i, s in enumerate(y == phn_to_st['!ENTER[2]']):
            if s and i - self._start_end[-1][0] > MIN_FRAMES_PER_SENTENCE:
                self._start_end[-1].append(i)
                self._start_end.append([i])
        self._start_end[-1].append(i+1)

    def _stackpad(self, start, end):
        """ Method because of the memoization. """
        if start in self._memoized_x and end in self._memoized_x[start]:
            return self._memoized_x[start][end]
        x = self._x[start:end]
        nf = self._nframes
        ret = numpy.zeros((x.shape[0], x.shape[1] * nf), dtype='float32')
        ba = (nf - 1) / 2  # before/after
        for i in xrange(x.shape[0]):
            ret[i] = numpy.pad(x[max(0, i - ba):i + ba +1].flatten(),
                    (max(0, (ba -i) * x.shape[1]),
                        max(0, ((i + ba + 1) - x.shape[0]) * x.shape[1])),
                    'constant', constant_values=(0, 0))
        self._memoized_x[start][end] = ret
        return ret

    def __iter__(self):
        for start, end in self._start_end:
            #yield shared(self._x[start:end], borrow=BORROW), shared(self._y[start:end], borrow=BORROW)
            if self._nframes > 1:
                #yield shared(self._stackpad(start, end)), shared(self._y[start:end])
                yield self._stackpad(start, end), self._y[start:end]
            else:
                yield self._x[start:end], self._y[start:end]


class DatasetIterator(object):
    """ An iterator over batches extracted from the dataset. """

    def __init__(self, x, y, batch_size):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        #self._n_batches = self._x.shape[0] / self._batch_size
        #self._rest = self._x.shape[0] % self._batch_size # TODO
        self._n_batches = self._x.get_value(borrow=BORROW).shape[0] / self._batch_size
        self._rest = self._x.get_value(borrow=BORROW).shape[0] % self._batch_size # TODO

    def __iter__(self):
        for index in xrange(self._n_batches):
            yield shared(self._x[index:index+self._batch_size], borrow=BORROW), shared(self._y[index:index+self._batch_size], borrow=BORROW)


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
                 hidden_layers_sizes=[1024, 1024], n_outs=62 * 3):
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

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.fmatrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
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
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.finetune_cost_sum = self.logLayer.negative_log_likelihood_sum(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, k):
        batch_x = T.fmatrix('batch_x')
        learning_rate = T.scalar('lr')  # learning rate to use

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            #markov_chain = shared(numpy.empty((batch_size, rbm.n_hidden), dtype='float32'), borrow=True)
            markov_chain = None
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=markov_chain, k=k)

            # compile the theano function
            fn = theano.function(inputs=[batch_x,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: batch_x})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for i, (param, gparam) in enumerate(zip(self.params, gparams)):
            updates[param] = param - gparam * learning_rate 

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.finetune_cost_sum,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def valid_and_test_scores(self, valid_set, test_set):
        """ Returns functions to get current validation and test scores. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x), theano.Param(batch_y)],
                outputs=self.errors,
                givens={self.x: batch_x, self.y: batch_y})

        # Create a function that scans the entire validation set
        def valid_score():
            return [score(batch_x, batch_y) for batch_x, batch_y in valid_set]

        # Create a function that scans the entire test set
        def test_score():
            return [score(batch_x, batch_y) for batch_x, batch_y in test_set]

        return valid_score, test_score


def test_DBN(finetune_lr=0.01, pretraining_epochs=0,
             pretrain_lr=0.01, k=1, training_epochs=100, # TODO 100+
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
    #datasets = load_data(dataset, nframes=N_FRAMES, features='fbank', scaling='normalize', cv_frac=0.2, speakers=False, numpy_array_only=True) 
    #datasets = load_data(dataset, nframes=N_FRAMES, features='fbank', scaling='student', cv_frac='fixed', speakers=False, numpy_array_only=True) 
    datasets = load_data(dataset, nframes=1, features='fbank', scaling='student', cv_frac='fixed', speakers=False, numpy_array_only=True) 

    train_set_x, train_set_y = datasets[0]  # if speakers, do test/test/test
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print "dataset loaded!"
    print "train set size", train_set_x.shape[0]
    print "validation set size", valid_set_x.shape[0]
    print "test set size", test_set_x.shape[0]

    to_int = {}
    with open('timit_to_int_and_to_state_dicts_tuple.pickle') as f:  # TODO
        to_int, _ = cPickle.load(f)
    train_set_iterator = DatasetSentencesIterator(train_set_x, train_set_y,
            to_int)
    valid_set_iterator = DatasetSentencesIterator(valid_set_x, valid_set_y,
            to_int)
    test_set_iterator = DatasetSentencesIterator(test_set_x, test_set_y,
            to_int)

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=train_set_x.shape[1],
              hidden_layers_sizes=[2496, 2496, 2496],
              n_outs=len(set(train_set_y)))#62 * 3)

    #########################
    # PRETRAINING THE MODEL #
    #########################
#    print '... getting the pretraining functions'
#    pretraining_fns = dbn.pretraining_functions(k=k)
#
#    print '... pre-training the model'
#    start_time = time.clock()
#    ## Pre-train layer-wise
#    #for i in xrange(dbn.n_layers): # TODO
#    for i in xrange(1):
#        # go through pretraining epochs
#        for epoch in xrange(pretraining_epochs):
#            # go through the training set
#            c = []
#            for batch_index, (batch_x, _) in enumerate(train_set_iterator):
#                tmp_lr = pretrain_lr / (1. + 0.05 * batch_index) # TODO
#                if i == 0:
#                    tmp_lr /= PRELEARNING_RATE_DENOMINATOR_FOR_GAUSSIAN
#                c.append(pretraining_fns[i](batch_x=batch_x, lr=tmp_lr))
#            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
#            print numpy.mean(c)
#        with open(output_file_name + '_layer_' + str(i) + '.pickle', 'w') as f:
#            cPickle.dump(dbn, f)
#        print "dumped a partially pre-trained model"
#
#    end_time = time.clock()
#    print >> sys.stderr, ('The pretraining code for file ' +
#                          os.path.split(__file__)[1] +
#                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    #with open('dbn_Gaussian_gpu_layer_2.pickle') as f:
    #    dbn = cPickle.load(f)

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn = dbn.get_SGD_trainer()
    validate_model, test_model = dbn.valid_and_test_scores(
            valid_set=valid_set_iterator, test_set=test_set_iterator)

    print '... finetuning the model'
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless TODO
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        train_set_iterator = DatasetSentencesIterator(train_set_x, 
                train_set_y, to_int)
        avg_costs = []
        for iteration, (x, y) in enumerate(train_set_iterator):
            avg_cost = train_fn(x, y, finetune_lr)
            avg_costs.append(avg_cost)
            #print('  epoch %i, sentence %i, '
            #'avg cost for this sentence %f' % \
            #      (epoch, iteration, avg_cost))
        print('  epoch %i, avg costs %f' % \
              (epoch, numpy.mean(avg_costs)))

        # we check the validation loss on every epoch
        validation_losses = validate_model()
        this_validation_loss = numpy.mean(validation_losses)  # TODO this is a mean of means (with different lengths)
        print('  epoch %i, validation error %f %%' % \
              (epoch, this_validation_loss * 100.))
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            with open(output_file_name + '.pickle', 'w') as f:
                cPickle.dump(dbn, f)
            # improve patience if loss improvement is good enough
            if (this_validation_loss < best_validation_loss *
                improvement_threshold):
                patience = max(patience, iteration * patience_increase)
            # save best validation score and iteration number
            best_validation_loss = this_validation_loss
            # test it on the test set
            test_losses = test_model()
            test_score = numpy.mean(test_losses)  # TODO this is a mean of means (with different lengths)
            print(('  epoch %i, test error of '
                   'best model %f %%') %
                  (epoch, test_score * 100.))
        if patience <= iteration:
            done_looping = True
            break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%, '
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    with open(output_file_name + '.pickle', 'w') as f:
        cPickle.dump(dbn, f)


if __name__ == '__main__':
    test_DBN()
