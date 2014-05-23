import cPickle
import os
import sys
import time
import socket

import numpy
from collections import OrderedDict, defaultdict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared

from logistic_regression import LogisticRegression 
from dataset_sentences_iterator import DatasetSentencesIterator
from relu_layer import ReLU, RecurrentReLU, DropoutReLU, dropout
from prep_timit import load_data

#DATASET = '/home/gsynnaeve/datasets/TIMIT'
#DATASET = '/media/bigdata/TIMIT'
#DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/wo_sa'
DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split'
if socket.gethostname() == "syhws-MacBook-Pro.local":
    DATASET = '/Users/gabrielsynnaeve/postdoc/datasets/TIMIT_train_dev_test'
N_FEATURES = 40  # filterbanks
N_FRAMES = 21  # HAS TO BE AN ODD NUMBER 
               #(same number before and after center frame)
IN_DROPOUT_RATE = 0.2
DEBUG_ON_TEST_ONLY = False
output_file_name = 'RRNN_dropout'


class RRNN(object):
    """Recurrent ReLU Neural Network
    """

    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=N_FEATURES * N_FRAMES,
            relu_layers_sizes=[1024, 1024, 1024],
            recurrent_connections=[2],  # layer(s), can only be i^t -> i^{t+1}
            n_outs=62 * 3,
            rho=0.9, eps=1.E-6):
        """ TODO 
        """

        self.relu_layers = []
        self.dropout_relu_layers = []
        self.params = []
        self.dropout_params = []
        self.n_layers = len(relu_layers_sizes)
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        self.n_outs = n_outs

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x')
        self.y = T.ivector('y')

        input_dropout_rate = IN_DROPOUT_RATE
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = relu_layers_sizes[i-1]

            if i == 0:
                layer_input = self.x
                dropout_layer_input = dropout(numpy_rng, self.x, p=input_dropout_rate)
            else:
                layer_input = self.relu_layers[-1].output
                dropout_layer_input = self.dropout_relu_layers[-1].output
                input_dropout_rate = self.dropout_relu_layers[-1].dropout_rate

            if i in recurrent_connections:
                # TODO
                inputr_size = relu_layers_sizes[i]
                previous_output = T.fmatrix('previous_output')
                relu_layer = RecurrentReLU(rng=numpy_rng,
                        input=layer_input, in_stack=previous_output,
                        n_in=input_size, n_in_stack=inputr_size,
                        n_out=inputr_size)
                #relu_layer.in_stack = relu_layer.output # TODO TODO TODO
                # /TODO
                self.params.extend(relu_layer.params)
                self._accugrads.extend([shared(value=numpy.zeros((n_ins, relu_layers_sizes[0]), dtype='float32'), name='accugrad_W', borrow=True), shared(value=numpy.zeros((relu_layers_sizes[0], ), dtype='float32'), name='accugrad_b', borrow=True), shared(value=numpy.zeros((n_outs, relu_layers_sizes[0]), dtype='float32'), name='accugrad_Ws', borrow=True)])
                self._accudeltas.extend([shared(value=numpy.zeros((n_ins, relu_layers_sizes[0]), dtype='float32'), name='accudelta_W', borrow=True), shared(value=numpy.zeros((relu_layers_sizes[0], ), dtype='float32'), name='accudelta_b', borrow=True), shared(value=numpy.zeros((n_outs, relu_layers_sizes[0]), dtype='float32'), name='accudelta_Ws', borrow=True)])

            else:
                dropout_relu_layer = DropoutReLU(rng=numpy_rng,
                        input=dropout_layer_input,
                        n_in=input_size,
                        n_out=relu_layers_sizes[i])
                relu_layer = ReLU(rng=numpy_rng,
                        input=layer_input,
                        n_in=input_size,
                        n_out=relu_layers_sizes[i],
                        W=dropout_relu_layer.W * (1 - input_dropout_rate),
                        b=dropout_relu_layer.b)

                self.dropout_params.extend(dropout_relu_layer.params)
                self.params.extend(relu_layer.params)

                self._accugrads.extend([shared(value=numpy.zeros((input_size, relu_layers_sizes[i]), dtype='float32'), name='accugrad_W', borrow=True), shared(value=numpy.zeros((relu_layers_sizes[i], ), dtype='float32'), name='accugrad_b', borrow=True)])
                self._accudeltas.extend([shared(value=numpy.zeros((input_size, relu_layers_sizes[i]), dtype='float32'), name='accudelta_W', borrow=True), shared(value=numpy.zeros((relu_layers_sizes[i], ), dtype='float32'), name='accudelta_b', borrow=True)])

            self.dropout_relu_layers.append(dropout_relu_layer)
            self.relu_layers.append(relu_layer)


        # We now need to add a logistic layer on top of the MLP
        self.dropout_logLayer = LogisticRegression(
            input=self.dropout_relu_layers[-1].output,
            n_in=relu_layers_sizes[-1],
            n_out=n_outs)
        self.logLayer = LogisticRegression(  # TODO check weights multiplication
            input=self.relu_layers[-1].output,
            n_in=relu_layers_sizes[-1],
            n_out=n_outs,
            W=self.dropout_logLayer.W * (1 - self.dropout_relu_layers[-1].dropout_rate),
            b=self.dropout_logLayer.b)
        self.dropout_params.extend(self.dropout_logLayer.params)
        self.params.extend(self.logLayer.params)
        self._accugrads.extend([shared(value=numpy.zeros((relu_layers_sizes[-1], n_outs), dtype='float32'), name='accugrad_W', borrow=True), shared(value=numpy.zeros((n_outs, ), dtype='float32'), name='accugrad_b', borrow=True)])
        self._accudeltas.extend([shared(value=numpy.zeros((relu_layers_sizes[-1], n_outs), dtype='float32'), name='accudelta_W', borrow=True), shared(value=numpy.zeros((n_outs, ), dtype='float32'), name='accudelta_b', borrow=True)])

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.dropout_finetune_cost = self.dropout_logLayer.negative_log_likelihood(self.y)
        self.dropout_finetune_cost_sum = self.dropout_logLayer.negative_log_likelihood_sum(self.y)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.finetune_cost_sum = self.logLayer.negative_log_likelihood_sum(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        cost = self.dropout_finetune_cost_sum
        params = self.dropout_params
        gparams = T.grad(cost, params)

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
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
        batch_y = T.ivector('batch_y')
        cost = self.dropout_finetune_cost_sum
        #cost = self.finetune_cost_sum TODO
        params = self.dropout_params
        gparams = T.grad(cost, params)

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y)],
            outputs=cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        cost = self.dropout_finetune_cost_sum
        params = self.dropout_params
        gparams = T.grad(cost, params)

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def score_classif(self, given_set):
        """ Returns functions to get current classification scores. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x), theano.Param(batch_y)],
                outputs=self.errors,
                givens={self.x: batch_x, self.y: batch_y})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref


def test_RRNN(finetune_lr=0.0001, pretraining_epochs=0,
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

    print "output file name:", output_file_name
    print "loading dataset from", dataset
    datasets = load_data(dataset, nframes=1, features='fbank', scaling='normalize', cv_frac='fixed', speakers=False, numpy_array_only=True) 

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print "dataset loaded!"
    print "train set size", train_set_x.shape[0]
    print "validation set size", valid_set_x.shape[0]
    print "test set size", test_set_x.shape[0]
    print "phones in train", len(set(train_set_y))
    print "phones in valid", len(set(valid_set_y))
    print "phones in test", len(set(test_set_y))

    to_int = {}
    with open('timit_to_int_and_to_state_dicts_tuple.pickle') as f:  # TODO
        to_int, _ = cPickle.load(f)
    train_set_iterator = DatasetSentencesIterator(train_set_x, train_set_y,
            to_int, N_FRAMES)
    valid_set_iterator = DatasetSentencesIterator(valid_set_x, valid_set_y,
            to_int, N_FRAMES)
    test_set_iterator = DatasetSentencesIterator(test_set_x, test_set_y,
            to_int, N_FRAMES)

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'

    nnet = RRNN(numpy_rng=numpy_rng, n_ins=N_FRAMES * N_FEATURES,
              relu_layers_sizes=[2400, 2400, 2400, 2400, 2400, 2400],
              recurrent_connections=[],
              n_outs=len(set(train_set_y)))

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn = nnet.get_adadelta_trainer()
    #train_fn = nnet.get_SGD_trainer()
    #train_fn = nnet.get_adagrad_trainer()

    train_scoref = nnet.score_classif(train_set_iterator)
    valid_scoref = nnet.score_classif(valid_set_iterator)
    test_scoref = nnet.score_classif(test_set_iterator)
    dataset_iterator = train_set_iterator

    if DEBUG_ON_TEST_ONLY:
        dataset_iterator = test_set_iterator
        train_scoref = test_scoref

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
        avg_costs = []
        for iteration, (x, y) in enumerate(dataset_iterator):
            avg_cost = train_fn(x, y)
            #avg_cost = train_fn(x, y, finetune_lr)
            avg_costs.append(avg_cost)
            #print('  epoch %i, sentence %i, '
            #'avg cost for this sentence %f' % \
            #      (epoch, iteration, avg_cost))
        #print('  epoch %i, avg costs %f, avg accudeltas %f' % \
        #      (epoch, numpy.mean(avg_costs), numpy.mean([T.mean(ad).eval() for ad in nnet._accudeltas])))
        print('  epoch %i, avg costs %f' % \
              (epoch, numpy.mean(avg_costs)))
        print('  epoch %i, training error %f %%' % \
              (epoch, numpy.mean(train_scoref()) * 100.))

        if DEBUG_ON_TEST_ONLY:
            continue

        # we check the validation loss on every epoch
        validation_losses = valid_scoref()
        this_validation_loss = numpy.mean(validation_losses)  # TODO this is a mean of means (with different lengths)
        print('  epoch %i, validation error %f %%' % \
              (epoch, this_validation_loss * 100.))
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            with open(output_file_name + '.pickle', 'w') as f:
                cPickle.dump(nnet, f)
            # improve patience if loss improvement is good enough
            if (this_validation_loss < best_validation_loss *
                improvement_threshold):
                patience = max(patience, iteration * patience_increase)
            # save best validation score and iteration number
            best_validation_loss = this_validation_loss
            # test it on the test set
            test_losses = test_scoref()
            test_score = numpy.mean(test_losses)  # TODO this is a mean of means (with different lengths)
            print(('  epoch %i, test error of '
                   'best model %f %%') %
                  (epoch, test_score * 100.))
        if patience <= iteration:  # TODO correct that
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
        cPickle.dump(nnet, f)


if __name__ == '__main__':
    test_RRNN()
    # TODO args
