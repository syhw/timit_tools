"""
"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared
from collections import OrderedDict

from logistic_timit import LogisticRegression 
from mlp import HiddenLayer
from rbm import RBM
from grbm import GRBM
from prep_timit import load_data

#DATASET = '/media/bigdata/TIMIT'
SPEAKERS = False
###DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT'
DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split'
#DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/std_split'
if SPEAKERS:
    DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT'
N_FRAMES = 13  # HAS TO BE AN ODD NUMBER 
               #(same number before and after center frame)
LEARNING_RATE_DENOMINATOR_FOR_GAUSSIAN = 50. # we take a lower learning rate
                                             # for the Gaussian RBM
output_file_name = 'dbn_analyze_timit'


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=39 * N_FRAMES,
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
        self.x = T.matrix('x')  # the data is presented as rasterized images
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

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

        # We now add all the logistic layers for all the RBM (to evaluate pretraining and 
        # where the errors comes from)
        self.layered_classifiers = [LogisticRegression(
            input=self.sigmoid_layers[i].output,
            n_in=hidden_layers_sizes[i],
            n_out=n_outs) for i in xrange(self.n_layers)]
#        self.layered_classifiers.append(LogisticRegression( TODO
#            input=self.x,
#            n_in=n_ins,
#            n_out=n_outs)) # classifier with outputs of all layers
        self.layered_classifiers.append(LogisticRegression(
            input=self.x,
            n_in=n_ins,
            n_out=n_outs)) # classifier from MFCC only (for comparison)
        self.layered_errors = [self.layered_classifiers[i].errors(self.y) 
                for i in xrange(len(self.layered_classifiers))]


    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

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
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.01)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                    train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    
    def pretraining_eval_function(self, layer, train_set_x, train_set_y,
            valid_set_x, valid_set_y, test_set_x=None, test_set_y=None,
            batch_size=20, n_epochs=10000):
        ''' Generates all `validation_error` fns that gives the current validation error
        of a softmax classifier trained on top of the hidden layers

        '''
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = None
        if test_set_x != None and test_set_y != None:
            n_test_batches = test_set_x.get_value(borrow=True).shape[0]
            n_test_batches /= batch_size
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = 0.01 # TODO try others

        # gradient
        cost = self.layered_classifiers[layer].negative_log_likelihood(self.y)
        g_W = T.grad(cost=cost, wrt=self.layered_classifiers[layer].W)
        g_b = T.grad(cost=cost, wrt=self.layered_classifiers[layer].b)
        # updates
        updates = OrderedDict([(self.layered_classifiers[layer].W, 
                    self.layered_classifiers[layer].W - learning_rate * g_W),
                (self.layered_classifiers[layer].b,
                    self.layered_classifiers[layer].b - learning_rate * g_b)])
        # compiling a Theano function
        train_logistic_regr_i = theano.function(inputs=[index],
                outputs=self.layered_classifiers[layer].negative_log_likelihood(self.y),
                  updates=updates,
                  givens={self.x: train_set_x[index * batch_size:
                                              (index + 1) * batch_size],
                          self.y: train_set_y[index * batch_size:
                                              (index + 1) * batch_size]})
        valid_error_i = theano.function([index], self.layered_errors[layer],
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
        test_error_i = None
        if test_set_x != None and test_set_y != None:
            test_error_i = theano.function([index], self.layered_errors[layer],
                  givens={self.x: test_set_x[index * batch_size:
                                              (index + 1) * batch_size],
                          self.y: test_set_y[index * batch_size:
                                              (index + 1) * batch_size]})


        def valid_error_fn():
            print 'training a LogisticRegression on top of layer', layer
            # gradient descent iterations
            patience = 10000
            patience_increase = 2
            improvement_threshold = 0.995
            best_valid_error = 2.
            iter = 0
            epoch = 0
            best_params = None
            while (epoch < n_epochs):
                epoch = epoch + 1
                for minibatch_index in xrange(n_train_batches):
                    minibatch_avg_cost = train_logistic_regr_i(minibatch_index)
                    iter = epoch * n_train_batches + minibatch_index
                valid_error = numpy.mean([valid_error_i(i) for i in xrange(n_valid_batches)])    
                if valid_error < best_valid_error:
                    if valid_error < best_valid_error * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_valid_error = valid_error
                    best_params = (self.layered_classifiers[layer].W,
                            self.layered_classifiers[layer].b)
                if patience <= iter:
                    break
            (self.layered_classifiers[layer].W, self.layered_classifiers[layer].b) = best_params
            test_error = 'not computed'
            if test_set_x != None and test_set_y != None:
                test_error = numpy.mean([test_error_i(i) for i in xrange(n_test_batches)])
            return (best_valid_error, test_error)

        return valid_error_fn


    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def train_DBN(finetune_lr=0.01, pretraining_epochs=100,
             pretrain_lr=0.001, k=1, training_epochs=200,
             dataset=DATASET, batch_size=100, dbn_load_from=''):
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
    ###datasets = load_data(dataset, nframes=N_FRAMES, features='MFCC', scaling='normalize', pca_whiten=False, cv_frac=0.2, dataset_name='TIMIT_wo_sa', speakers=SPEAKERS)
    datasets = load_data(dataset, nframes=N_FRAMES, features='MFCC', scaling='normalize', pca_whiten=False, cv_frac='fixed', dataset_name='TIMIT_train_dev_test', speakers=SPEAKERS)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1] 
    test_set_x, test_set_y = datasets[2]
    N_OUTS = 62 * 3 # #phones * #states
    if SPEAKERS:
        from collections import Counter
        c = Counter(train_set_y.eval())
        N_OUTS = len(c)
    print "dataset loaded!"
    print "train set size", train_set_x.shape[0]
    print "validation set size", valid_set_x.shape[0]
    print "test set size", test_set_x.shape[0]
    print "N_OUTS:", N_OUTS

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    print "train_set_x.shape.eval()", train_set_x.shape.eval()
    assert(train_set_x.shape[1].eval() == N_FRAMES * 39) # check
    dbn = DBN(numpy_rng=numpy_rng, n_ins=train_set_x.shape[1].eval(),
              hidden_layers_sizes=[2496, 2496, 2496],
              n_outs=N_OUTS)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... evaluating on MFCC only, error rate of a LogisticRegression:'
    on_top_of_MFCC_fn = dbn.pretraining_eval_function(layer=-1,
                                                train_set_x=train_set_x,
                                                train_set_y=train_set_y,
                                                valid_set_x=valid_set_x,
                                                valid_set_y=valid_set_y,
                                                test_set_x=test_set_x,
                                                test_set_y=test_set_y,
                                                batch_size=batch_size)

    print 'error rate:', on_top_of_MFCC_fn()
    #dbn = None ### TOREMOVE
    #with open('dbn_analyze_timit__plr1.0E-03_pep100_flr1.0E-03_fep_10_k1_layer_1.pickle') as f: ### TOREMOVE
    #    dbn = cPickle.load(f) ### TOREMOVE

    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)
    pretraining_eval_fns = [dbn.pretraining_eval_function(layer=ii,
                                                train_set_x=train_set_x,
                                                train_set_y=train_set_y,
                                                valid_set_x=valid_set_x,
                                                valid_set_y=valid_set_y,
                                                test_set_x=test_set_x,
                                                test_set_y=test_set_y,
                                                batch_size=batch_size)
                                        for ii in xrange(dbn.n_layers)]
    for i in xrange(dbn.n_layers):
        print i, pretraining_eval_fns[i]()

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        #######################
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                tmp_lr = pretrain_lr / (1. + 0.05 * batch_index) # TODO
                if i == 0:
                    tmp_lr /= LEARNING_RATE_DENOMINATOR_FOR_GAUSSIAN
                c.append(pretraining_fns[i](index=batch_index, lr=tmp_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
            ##############################
            print('>>> (cross_val, test) error rates of LogisticRegression on top of the hidden layer %d is' % i)
            print (pretraining_eval_fns[i]())
            # TODO stop pretraining when this error rate goes up (early stopping)
            ##############################
        with open(output_file_name + '_layer_' + str(i) + '.pickle', 'w') as f:
            cPickle.dump(dbn, f)
        print "dumped a partially pre-trained model"
        #######################

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    if dbn_load_from != '':
        with open(dbn_load_from) as f:
            dbn = cPickle.load(f)
        print 'loaded this dbn:', dbn_load_from
    #with open(output_file_name + '_layer_2.pickle') as f:
    #    dbn = cPickle.load(f)

    #datasets = load_data(dataset, nframes=N_FRAMES, unit=False, student=True, pca_whiten=False, cv_frac=0.2, dataset_name='TIMIT', speakers=SPEAKERS)
    #train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = None, None, None, None, None, None
    #train_set_x, train_set_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1] 
    #test_set_x, test_set_y = datasets[2]
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    print "number of training (fine-tuning) batches", n_train_batches
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                ##############################
                for layer_ind in xrange(dbn.n_layers):
                    print('>>> (cross-val, test) error rate of a LogisticRegression on top of layer %d is' % layer_ind)
                    print(pretraining_eval_fns[layer_ind]())
                ##############################

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    with open(output_file_name + '.pickle', 'w') as f:
                        cPickle.dump(dbn, f)

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    with open(output_file_name + '.pickle', 'w') as f:
        cPickle.dump(dbn, f)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        usage = """usage: python DBN_analyze_timit $pretrain_lr $pretrain_epochs $finetune_lr $finetune_epochs $k(for_CD) [dbn_load_from.pickle]"""
        print usage
        load_from = ''
        if len(sys.argv) > 6:
            load_from = sys.argv[6]
        plr, pep, flr, fep, k = sys.argv[1:6]
        plr = float(plr)
        pep = int(pep)
        flr = float(flr)
        fep = int(fep)
        k = int(k)
        tmp = '_plr%.1E_pep%d_flr%.1E_fep%d_k%d' % (plr, pep, flr, fep, k)
        output_file_name = output_file_name + tmp
        train_DBN(finetune_lr=flr, pretraining_epochs=pep,
             pretrain_lr=plr, k=k, training_epochs=fep,
             dataset=DATASET, batch_size=100, dbn_load_from=load_from)
    else:
        train_DBN()
