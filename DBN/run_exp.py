"""Runs deep learning experiments on speech dataset.

Usage:
    run_exp.py [--dataset=path] [--dataset-name=timit] 
    [--iterator-type=sentences] [--batch-size=100] [--nframes=13] 
    [--features=fbank] [--init-lr=0.001] [--epochs=500] 
    [--network-type=dropout_XXX] [--trainer-type=adadelta] 
    [--prefix-output-fname=my_prefix_42] [--debug-test] [--debug-print] 
    [--debug-time]


Options:
    -h --help                   Show this screen
    --version                   Show version
    --dataset=str               A valid path to the dataset
    default is timit
    --dataset-name=str          Name of the dataset (for outputs/saves)
    default is "timit"
    --iterator-type=str         "sentences" | "batch" | "dtw"
    default is "sentences"
    --batch-size=int            Batch size, used only by the batch iterator
    default is 100 (unused for "sentences" iterator type)
    --nframes=int               Number of frames to base the first layer on
    default is 13
    --features=str              "fbank" | "MFCC" (some others are not tested)
    default is "fbank"
    --init-lr=float             Initial learning rate for SGD
    default is 0.001 (that is very low intentionally)
    --epochs=int                Max number of epochs (always early stopping)
    default is 500
    --network-type=str         "dropout*" | "*" | "*ab_net*"
    default is "dropout_XXX"
    --trainer-type=str         "SGD" | "adagrad" | "adadelta"
    default is "adadelta"
    --prefix-output-fname=str  An additional prefix to the output file name
    default is "" (empty string)
    --debug-test               Flag that activates training on the test set
    default is False, using it makes it True
    --debug-print              Flag that activates printing the symbolic expr.
                               computed by the network
    default is False, using it makes it True
    --debug-time               Flag that activates timing epoch duration
    default is False, using it makes it True

"""

import socket, docopt, cPickle, time, sys, os
import numpy

from prep_timit import load_data
from dataset_iterators import DatasetSentencesIterator, DatasetBatchIterator
from dataset_iterators import DatasetDTWIterator
from layers import Linear, ReLU 
from classifiers import LogisticRegression
from nnet_archs import NeuralNet, DropoutNet, ABNeuralNet

DEFAULT_DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split'
if socket.gethostname() == "syhws-MacBook-Pro.local":
    DEFAULT_DATASET = '/Users/gabrielsynnaeve/postdoc/datasets/TIMIT_train_dev_test'
elif socket.gethostname() == "TODO":  # TODO
    DEFAULT_DATASET = '/media/bigdata/TIMIT_train_dev_test'


def run(dataset=DEFAULT_DATASET, dataset_name='timit',
        iterator_type=DatasetSentencesIterator, batch_size=100,
        nframes=13, features="fbank",
        init_lr=0.001, max_epochs=500, 
        network_type="dropout_XXX", trainer_type="adadelta",
        layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
        layers_sizes=[2400, 2400, 2400, 2400],
        dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
        recurrent_connections=[],
        prefix_fname='',
        debug_on_test_only=False,
        debug_print=False,
        debug_time=False):
    """
    FIXME TODO
    """

    output_file_name = dataset_name
    if prefix_fname != "":
        output_file_name = prefix_fname + "_" + dataset_name
    output_file_name += "_" + features + str(nframes)
    output_file_name += "_" + network_type + "_" + trainer_type
    print "output file name:", output_file_name

    n_ins = None
    n_outs = None
    print "loading dataset from", dataset
     # TODO DO A FUNCTION
    if dataset[-7:] == '.joblib':
        # TODO
        import joblib
        datasets = joblib.load(dataset)
        from random import shuffle
        #datasets = [(word_label, fbanks1, fbanks2, DTW_cost, DTW_1to2, DTW_2to1)]
        all_the_data = numpy.r_[numpy.concatenate([e[1] for e in datasets]),
            numpy.concatenate([e[2] for e in datasets])]
        mean = numpy.mean(all_the_data, 0)
        std = numpy.std(all_the_data, 0)
        data = [((e[1][e[-2]] - mean) / std, (e[2][e[-1]] - mean) / std)
                for e in datasets]
        shuffle(data)
        x1, x2 = zip(*data)
        y = [1 for _ in xrange(len(data))]
        assert x1[0].shape[0] == x2[0].shape[0]
        assert x1[0].shape[1] == x2[0].shape[1]
        assert len(x1) == len(x2)
        assert len(x1) == len(y)
        ten_percent = int(0.1 * len(data))

        n_ins = x1[0].shape[1] * nframes
        n_outs = 50

        # TODO
        print "nframes:", nframes
        train_set_iterator = iterator_type(x1[:-ten_percent], 
                x2[:-ten_percent], y[:-ten_percent], # TODO
                nframes=nframes, batch_size=batch_size, margin=True)
        valid_set_iterator = iterator_type(x1[-ten_percent:], 
                x2[-ten_percent:], y[-ten_percent:],  # TODO
                nframes=nframes, batch_size=batch_size, margin=True)
        test_set_iterator = iterator_type(x1[-ten_percent:], 
                x2[-ten_percent:], y[-ten_percent:], # TODO
                nframes=nframes, batch_size=batch_size, margin=True)
    else:
        datasets = load_data(dataset, nframes=1, features=features, scaling='normalize', cv_frac='fixed', speakers=False, numpy_array_only=True) 

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        assert train_set_x.shape[1] == valid_set_x.shape[1]
        assert test_set_x.shape[1] == valid_set_x.shape[1]

        print "dataset loaded!"
        print "train set size", train_set_x.shape[0]
        print "validation set size", valid_set_x.shape[0]
        print "test set size", test_set_x.shape[0]
        print "phones in train", len(set(train_set_y))
        print "phones in valid", len(set(valid_set_y))
        print "phones in test", len(set(test_set_y))
        n_outs = len(set(train_set_y))

        to_int = {}
        with open(dataset_name + '_to_int_and_to_state_dicts_tuple.pickle') as f:
            to_int, _ = cPickle.load(f)

        print "nframes:", nframes
        train_set_iterator = iterator_type(train_set_x, train_set_y,
                to_int, nframes=nframes, batch_size=batch_size)
        valid_set_iterator = iterator_type(valid_set_x, valid_set_y,
                to_int, nframes=nframes, batch_size=batch_size)
        test_set_iterator = iterator_type(test_set_x, test_set_y,
                to_int, nframes=nframes, batch_size=batch_size)
        n_ins = test_set_x.shape[1]*nframes

    assert n_ins != None
    assert n_outs != None

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'

    # TODO the proper network type other than just dropout or not
    nnet = None
    if "dropout" in network_type:
        nnet = DropoutNet(numpy_rng=numpy_rng, 
                n_ins=n_ins,
                layers_types=layers_types,
                layers_sizes=layers_sizes,
                dropout_rates=dropout_rates,
                n_outs=n_outs,
                debugprint=debug_print)
    elif "ab_net" in network_type:
        nnet = ABNeuralNet(numpy_rng=numpy_rng, 
                n_ins=n_ins,
                layers_types=layers_types,
                layers_sizes=layers_sizes,
                n_outs=n_outs,
                debugprint=debug_print)
    else:
        nnet = NeuralNet(numpy_rng=numpy_rng, 
                n_ins=n_ins,
                layers_types=layers_types,
                layers_sizes=layers_sizes,
                n_outs=n_outs,
                debugprint=debug_print)

    # get the training, validation and testing function for the model
    print '... getting the training functions'
    print trainer_type
    train_fn = None
    if trainer_type == "adadelta":
        train_fn = nnet.get_adadelta_trainer()
    elif trainer_type == "adagrad":
        train_fn = nnet.get_adagrad_trainer()
    else:
        train_fn = nnet.get_SGD_trainer()

    train_scoref = nnet.score_classif(train_set_iterator)
    valid_scoref = nnet.score_classif(valid_set_iterator)
    test_scoref = nnet.score_classif(test_set_iterator)
    data_iterator = train_set_iterator

    if debug_on_test_only:
        data_iterator = test_set_iterator
        train_scoref = test_scoref

    print '... training the model'
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
    lr = init_lr
    timer = None

    while (epoch < max_epochs) and (not done_looping):
        epoch = epoch + 1
        avg_costs = []
        if debug_time:
            timer = time.time()
        for iteration, (x, y) in enumerate(data_iterator):
            avg_cost = 0.
            if "delta" in trainer_type:
                print x, y
                avg_cost = train_fn(x, y)
            else:
                avg_cost = train_fn(x, y, lr)
            avg_costs.append(avg_cost)
        if debug_time:
            print('  epoch %i took %f seconds' % (epoch, time.time() - timer))
        print('  epoch %i, avg costs %f' % \
              (epoch, numpy.mean(avg_costs)))
        print('  epoch %i, training error %f %%' % \
              (epoch, numpy.mean(train_scoref()) * 100.))
        # TODO update lr(t) = lr(0) / (1 + lr(0) * lambda * t)
        # or another scheme for learning rate decay

        if debug_on_test_only:
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

if __name__=='__main__':
    arguments = docopt.docopt(__doc__, version='run_exp version 0.1')
    dataset=DEFAULT_DATASET
    if arguments['--dataset'] != None:
        dataset = arguments['--dataset']
    dataset_name = 'timit'
    if arguments['--dataset-name'] != None:
        dataset_name = arguments['--dataset-name']
    iterator_type = DatasetSentencesIterator
    if arguments['--iterator-type'] != None:
        if "sentences" in arguments['--iterator-type']:
            iterator_type = DatasetSentencesIterator
        elif "dtw" in arguments['--iterator-type']:
            iterator_type = DatasetDTWIterator
        else:
            iterator_type = DatasetBatchIterator  # TODO
    batch_size = 100
    if arguments['--batch-size'] != None:
        batch_size = int(arguments['--batch-size'])
    nframes = 13
    if arguments['--nframes'] != None:
        nframes = int(arguments['--nframes'])
    features = 'fbank'
    if arguments['--features'] != None:
        features = arguments['--features']
    init_lr = 0.001
    if arguments['--init-lr'] != None:
        init_lr = float(arguments['--init-lr'])
    max_epochs = 500
    if arguments['--epochs'] != None:
        max_epochs = int(arguments['--epochs'])
    network_type = 'dropout_XXX'
    if arguments['--network-type'] != None:
        network_type = arguments['--network-type']
    trainer_type = 'adadelta'
    if arguments['--trainer-type'] != None:
        trainer_type = arguments['--trainer-type']
    prefix_fname = ''
    if arguments['--prefix-output-fname'] != None:
        prefix_fname = arguments['--prefix-output-fname']
    debug_on_test_only = False
    if arguments['--debug-test']:
        debug_on_test_only = True
    debug_print = False
    if arguments['--debug-print']:
        debug_print = True
    debug_time = False
    if arguments['--debug-time']:
        debug_time = True

    run(dataset=dataset, dataset_name=dataset_name,
        iterator_type=iterator_type, batch_size=batch_size,
        nframes=nframes, features=features,
        init_lr=init_lr, max_epochs=max_epochs, 
        network_type=network_type, trainer_type=trainer_type,
        #layers_types=[Linear, ReLU, ReLU, LogisticRegression],
        #layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
        #layers_sizes=[1024, 1024, 1024],  # TODO in opts
        #dropout_rates=[0.2, 0.3, 0.4, 0.5],  # TODO in opts
        layers_types=[ReLU, ReLU, ReLU],
        layers_sizes=[1024, 1024],  # TODO in opts
        #layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
        #layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
        #layers_sizes=[2000, 2000, 2000, 2000],  # TODO in opts
        #dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],  # TODO in opts
        recurrent_connections=[],  # TODO in opts
        prefix_fname=prefix_fname,
        debug_on_test_only=debug_on_test_only,
        debug_print=debug_print,
        debug_time=debug_time)
