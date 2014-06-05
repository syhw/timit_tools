"""Runs deep learning experiments on speech dataset.

Usage:
    run_exp.py [--dataset-path=path] [--dataset-name=timit] 
    [--iterator-type=sentences] [--batch-size=100] [--nframes=13] 
    [--features=fbank] [--init-lr=0.001] [--epochs=500] 
    [--network-type=dropout_XXX] [--trainer-type=adadelta] 
    [--prefix-output-fname=my_prefix_42] [--debug-test] [--debug-print=lvl] 
    [--debug-time] [--debug-plot=0]


Options:
    -h --help                   Show this screen
    --version                   Show version
    --dataset-path=str          A valid path to the dataset
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
    --debug-print=int          Level of debug printing. 0: nothing, 1: network
    default is 0               2: epochs/iters related
    default is False, using it makes it True
    --debug-time               Flag that activates timing epoch duration
    default is False, using it makes it True
    --debug-plot=int           Level of debug plotting, 1: costs
    default is 0               >= 2: gradients & updates
"""

import socket, docopt, cPickle, time, sys, os
import numpy
import prettyplotlib as ppl
import matplotlib.pyplot as plt
import joblib
import random
from random import shuffle

from prep_timit import load_data
from dataset_iterators import DatasetSentencesIterator, DatasetBatchIterator
from dataset_iterators import DatasetDTWIterator
from layers import Linear, ReLU, SigmoidLayer
from classifiers import LogisticRegression
from nnet_archs import NeuralNet, DropoutNet, ABNeuralNet

DEFAULT_DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split'
if socket.gethostname() == "syhws-MacBook-Pro.local":
    DEFAULT_DATASET = '/Users/gabrielsynnaeve/postdoc/datasets/TIMIT_train_dev_test'
elif socket.gethostname() == "TODO":  # TODO
    DEFAULT_DATASET = '/media/bigdata/TIMIT_train_dev_test'
DEBUG = False

OLD_DTW_DATA = False


def print_mean_weights_biases(params):
    for layer_ind, param in enumerate(params):
        filler = "weight"
        if layer_ind % 2:
            filler = "bias"
        print("layer %i mean %s values %f and std devs %f" % (layer_ind/2, 
            filler, numpy.mean(param.eval()), numpy.std(param.eval())))


def plot_costs(cost):
    # TODO
    pass


def rolling_avg_pgu(iteration, pgu, l):
    # (iteration * pgu + l) / (iteration + 1)
    assert len(l) == len(pgu)
    ll = len(l)/3
    params, gparams, updates = l[:ll], l[ll:-ll], l[-ll:]
    mpars, mgpars, mupds = pgu[:ll], pgu[ll:-ll], pgu[-ll:]
    ii = iteration + 1
    return [(iteration * mpars[k] + p) / ii for k, p in enumerate(params)] +\
            [(iteration * mgpars[k] + g) / ii for k, g in enumerate(gparams)] +\
            [(iteration * mupds[k] + u) / ii for k, u in enumerate(updates)]


def plot_params_gradients_updates(n, l):
    # TODO currently works only with THEANO_FLAGS="device=cpu" (not working on
    #CudaNDArrays)
    def plot_helper(li, ti, p):
        fig, ax = plt.subplots(1)
        if li % 2:
            title = "biases" + ti
            ppl.bar(ax, numpy.arange(p.shape[0]), p)
        else:
            title = "weights" + ti
            ppl.pcolormesh(fig, ax, p)
        plt.title(title)
        plt.savefig(title + ".png")
        #ppl.show()
        plt.close()
    ll = len(l)/3
    params, gparams, updates = l[:ll], l[ll:-ll], l[-ll:]
    if DEBUG:
        print "params"
        print params
        print "===================="
        print "gparams"  # TODO find out why not CudaNDArray here
        print gparams
        print "===================="
        print "updates"  # TODO find out why not CudaNDArray here
        print updates
    title_iter =  "_%04i" % n
    for layer_ind, param in enumerate(params):
        title = "_for_layer_" + str(layer_ind/3) + title_iter
        plot_helper(layer_ind, title, param)
    for layer_ind, gparam in enumerate(gparams):
        title = "_gradients_for_layer_" + str(layer_ind/3) + title_iter
        plot_helper(layer_ind, title, gparam)
    for layer_ind, update in enumerate(updates):
        title = "_updates_for_layer_" + str(layer_ind/3) + title_iter
        plot_helper(layer_ind, title, update)


def run(dataset_path=DEFAULT_DATASET, dataset_name='timit',
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
        debug_print=0,
        debug_time=False,
        debug_plot=0):
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
    print "loading dataset from", dataset_path
     # TODO DO A FUNCTION
    if dataset_path[-7:] == '.joblib':
        if OLD_DTW_DATA:
            data_same = joblib.load(dataset_path)
            #data_same = [(word_label, fbanks1, fbanks2, DTW_cost, DTW_1to2, DTW_2to1)]
            if debug_print:
                # some stats on the DTW
                dtw_costs = zip(*data_same)[3]
                words_frames = numpy.asarray([fb.shape[0] for fb in zip(*data_same)[1]])
                print "mean DTW cost", numpy.mean(dtw_costs), "std dev", numpy.std(dtw_costs)
                print "mean word length in frames", numpy.mean(words_frames), "std dev", numpy.std(words_frames)
                print "mean DTW cost per frame", numpy.mean(dtw_costs/words_frames), "std dev", numpy.std(dtw_costs/words_frames)
                # /some stats on the DTW
            # TODO maybe ceil on the DTW cost to be considered "same"

            x_arr_same = numpy.r_[numpy.concatenate([e[1] for e in data_same]),
                numpy.concatenate([e[2] for e in data_same])]
            print x_arr_same.shape

            # we need about as much negative examples as positive ones
            # TODO wrap this in try except or if
            tmp = dataset_path.split('/')
            neg_data_path = "/".join(tmp[:-1]) + "/neg" + tmp[-1][3:]
            data_diff = joblib.load(neg_data_path)
            x_arr_diff = numpy.r_[numpy.concatenate([e[0] for e in data_diff]),
                    numpy.concatenate([e[1] for e in data_diff])]
            print x_arr_diff.shape
            x_arr_all = numpy.concatenate([x_arr_same, x_arr_diff])
            mean = numpy.mean(x_arr_all, 0)
            std = numpy.std(x_arr_all, 0)
            numpy.savez("mean_std", mean=mean, std=std)

            x_same = [((e[1][e[-2]] - mean) / std, (e[2][e[-1]] - mean) / std)
                    for e in data_same]
            shuffle(x_same)  # in place
            y_same = [[1 for _ in xrange(len(e[0]))] for i, e in enumerate(x_same)]

            x_diff = [((e[0] - mean) / std, (e[1] - mean) / std)
                    for e in data_diff]
            shuffle(x_diff)
            y_diff = [[0 for _ in xrange(len(e[0]))] for i, e in enumerate(x_diff)]
            y = [j for i in zip(y_same, y_diff) for j in i]
            x = [j for i in zip(x_same, x_diff) for j in i]

        else:
            data_same = joblib.load(dataset_path)
            #data_same = [(word_label, talker1, talker2, fbanks1, fbanks2, DTW_cost, DTW_1to2, DTW_2to1)]
            if debug_print:
                # some stats on the DTW
                dtw_costs = zip(*data_same)[5]
                words_frames = numpy.asarray([fb.shape[0] for fb in zip(*data_same)[3]])
                print "mean DTW cost", numpy.mean(dtw_costs), "std dev", numpy.std(dtw_costs)
                print "mean word length in frames", numpy.mean(words_frames), "std dev", numpy.std(words_frames)
                print "mean DTW cost per frame", numpy.mean(dtw_costs/words_frames), "std dev", numpy.std(dtw_costs/words_frames)
            x_arr_same = numpy.r_[numpy.concatenate([e[3] for e in data_same]),
                numpy.concatenate([e[4] for e in data_same])]
            print x_arr_same.shape

            # generate data_diff:
#            spkr_words = {}
            same_spkr = 0
            for i, tup in enumerate(data_same):
#                spkr_words[tup[1]].append((i, 0))
#                spkr_words[tup[2]].append((i, 1))
                if tup[1] == tup[2]:
                    same_spkr += 1
#            to_del = []
#            for spkr, words in spkr_words.iteritems():
#                if len(words) < 2:
#                    to_del.append(spkr)
#            print "to del len:", len(to_del)
#            for td in to_del:
#                del spkr_words[td]
            ratio = same_spkr * 1. / len(data_same)
            print "ratio same spkr / all for same:", ratio
            data_diff = []
#            keys = spkr_words.keys()
#            lkeys = len(keys) - 1
            ldata_same = len(data_same)-1
            same_spkr_diff = 0
            for i in xrange(len(data_same)):
                word_1 = random.randint(0, ldata_same)
                word_1_type = data_same[word_1][0]
                word_2 = random.randint(0, ldata_same)
                while data_same[word_2][0] == word_1_type:
                    word_2 = random.randint(0, ldata_same)

                wt1 = random.randint(0, 1)
                wt2 = random.randint(0, 1)
                if data_same[word_1][1+wt1] == data_same[word_2][1+wt2]:
                    same_spkr_diff += 1
                pair = (data_same[word_1][3+wt1], data_same[word_2][3+wt2])
                data_diff.append(pair)

#                first_spkr = random.randint(0, lkeys)
#                possible_word_1 = spkr_words[keys[first_spkr]]
#                first_word = random.randint(0, len(possible_word_1) - 1)
#                if random.random() < ratio:
#                    # pick 2 different words same speaker
#                    second_word = random.randint(0, len(possible_word_1) - 1)
#                    while second_word == first_word:
#                        second_word = random.randint(0, len(possible_word_1) - 1)
#                    word1 = possible_word_1[first_word]
#                    word2 = possible_word_1[second_word]
#                    data_diff.append((data_same[
#                else:
#                    second_spkr = random.randint(0, lkeys)
#                    while second_spkr == first_spkr:
#                        second_spkr = random.randint(0, lkeys)
#                    possible_words = spkr_words[keys[first_spkr]]
#                    # pick 2 different words diff speakers

            ratio = same_spkr_diff * 1. / len(data_diff)
            print "ratio same spkr / all for diff:", ratio

            x_arr_diff = numpy.r_[numpy.concatenate([e[0] for e in data_diff]),
                    numpy.concatenate([e[1] for e in data_diff])]
            print x_arr_diff.shape

            x_arr_all = numpy.concatenate([x_arr_same, x_arr_diff])
            mean = numpy.mean(x_arr_all, 0)
            std = numpy.std(x_arr_all, 0)
            numpy.savez("mean_std_2", mean=mean, std=std)

            x_same = [((e[3][e[-2]] - mean) / std, (e[4][e[-1]] - mean) / std)
                    for e in data_same]
            shuffle(x_same)  # in place
            y_same = [[1 for _ in xrange(len(e[0]))] for i, e in enumerate(x_same)]
            x_diff = [((e[0] - mean) / std, (e[1] - mean) / std)
                    for e in data_diff]
            #shuffle(x_diff)
            y_diff = [[0 for _ in xrange(len(e[0]))] for i, e in enumerate(x_diff)]
            y = [j for i in zip(y_same, y_diff) for j in i]
            x = [j for i in zip(x_same, x_diff) for j in i]

        x1, x2 = zip(*x)
        assert x1[0].shape[0] == x2[0].shape[0]
        assert x1[0].shape[1] == x2[0].shape[1]
        assert len(x1) == len(x2)
        assert len(x1) == len(y)
        ten_percent = int(0.1 * len(x1))

        n_ins = x1[0].shape[1] * nframes
        n_outs = 100 # TODO

        print "nframes:", nframes
        train_set_iterator = iterator_type(x1[:-ten_percent], 
                x2[:-ten_percent], y[:-ten_percent], # TODO
                nframes=nframes, batch_size=batch_size, marginf=3) # TODO margin pass this 3 along before
        valid_set_iterator = iterator_type(x1[-ten_percent:], 
                x2[-ten_percent:], y[-ten_percent:],  # TODO
                nframes=nframes, batch_size=batch_size, marginf=3)

        ### TEST SET

        if OLD_DTW_DATA:
            test_dataset_path = "/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split/dtw_words_test.joblib"
            data_same = joblib.load(test_dataset_path)
            x_arr_same = numpy.r_[numpy.concatenate([e[1] for e in data_same]),
                numpy.concatenate([e[2] for e in data_same])]
            print x_arr_same.shape
            tmp = test_dataset_path.split('/')
            neg_data_path = "/".join(tmp[:-1]) + "/neg" + tmp[-1][3:]
            data_diff = joblib.load(neg_data_path)
            x_arr_diff = numpy.r_[numpy.concatenate([e[0] for e in data_diff]),
                    numpy.concatenate([e[1] for e in data_diff])]
            print x_arr_diff.shape
            x_arr_all = numpy.concatenate([x_arr_same, x_arr_diff])
            mean = numpy.mean(x_arr_all, 0)
            std = numpy.std(x_arr_all, 0)

            x_same = [((e[1][e[-2]] - mean) / std, (e[2][e[-1]] - mean) / std)
                    for e in data_same]
            shuffle(x_same)  # in place
            y_same = [[1 for _ in xrange(len(e[0]))] for i, e in enumerate(x_same)]

            x_diff = [((e[0] - mean) / std, (e[1] - mean) / std)
                    for e in data_diff]
            shuffle(x_diff)
            y_diff = [[0 for _ in xrange(len(e[0]))] for i, e in enumerate(x_diff)]
            y = [j for i in zip(y_same, y_diff) for j in i]
            x = [j for i in zip(x_same, x_diff) for j in i]
            
        else:
            test_dataset_path = "./dtw_words_2_dev.joblib"
            data_same = joblib.load(test_dataset_path)
            # DO ONLY SAME
            x_arr_same = numpy.r_[numpy.concatenate([e[3] for e in data_same]),
                numpy.concatenate([e[4] for e in data_same])]
            print x_arr_same.shape
            x_same = [((e[3][e[-2]] - mean) / std, (e[4][e[-1]] - mean) / std)
                    for e in data_same]
            shuffle(x_same)  # in place
            y_same = [[1 for _ in xrange(len(e[0]))] for i, e in enumerate(x_same)]
            x = x_same
            y = y_same

        x1, x2 = zip(*x)
        test_set_iterator = iterator_type(x1, x2, y,
            nframes=nframes, batch_size=batch_size, marginf=3)

    else:
        data = load_data(dataset_path, nframes=1, features=features, scaling='normalize', cv_frac='fixed', speakers=False, numpy_array_only=True) 

        train_set_x, train_set_y = data[0]
        valid_set_x, valid_set_y = data[1]
        test_set_x, test_set_y = data[2]
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
    print "Created a neural net as:",
    print str(nnet)

    # get the training, validation and testing function for the model
    print '... getting the training functions'
    print trainer_type
    train_fn = None
    if debug_plot or debug_print:
        if trainer_type == "adadelta":
            train_fn = nnet.get_adadelta_trainer(debug=True)
        elif trainer_type == "adagrad":
            train_fn = nnet.get_adagrad_trainer(debug=True)
        else:
            train_fn = nnet.get_SGD_trainer(debug=True)
    else:
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
    if debug_plot:
        print_mean_weights_biases(nnet.params)
    #with open(output_file_name + 'epoch_0.pickle', 'wb') as f:
    #    cPickle.dump(nnet, f)

    while (epoch < max_epochs) and (not done_looping):
        epoch = epoch + 1
        avg_costs = []
        avg_params_gradients_updates = []
        if debug_time:
            timer = time.time()
        for iteration, (x, y) in enumerate(data_iterator):
            avg_cost = 0.
            if "ab_net" in network_type:  # remove need for this if
                if "delta" in trainer_type:  # TODO remove need for this if
                    avg_cost = train_fn(x[0], x[1], y)
                else:
                    avg_cost = train_fn(x[0], x[1], y, lr)
                if debug_print >= 3:
                    print "cost:", avg_cost[0]
                if debug_plot >= 2:
                    plot_costs(avg_cost[0])
                    if not len(avg_params_gradients_updates):
                        avg_params_gradients_updates = avg_cost[1:]
                    else:
                        avg_params_gradients_updates = rolling_avg_pgu(
                                iteration, avg_params_gradients_updates,
                                avg_cost[1:])
                if debug_plot >= 3:
                    plot_params_gradients_updates(iteration, avg_cost[1:])
            else:
                if "delta" in trainer_type:  # TODO remove need for this if
                    avg_cost = train_fn(x, y)
                else:
                    avg_cost = train_fn(x, y, lr)
            if type(avg_cost) == list:
                avg_costs.append(avg_cost[0])
            else:
                avg_costs.append(avg_cost)
        if debug_print >= 2:
            print_mean_weights_biases(nnet.params)
        if debug_plot >= 2:
            plot_params_gradients_updates(epoch, avg_params_gradients_updates)
        if debug_time:
            print('  epoch %i took %f seconds' % (epoch, time.time() - timer))
        print('  epoch %i, avg costs %f' % \
              (epoch, numpy.mean(avg_costs)))
        print('  epoch %i, training error %f' % \
              (epoch, numpy.mean(train_scoref())))
        # TODO update lr(t) = lr(0) / (1 + lr(0) * lambda * t)
        # or another scheme for learning rate decay
        #with open(output_file_name + 'epoch_' +str(epoch) + '.pickle', 'wb') as f:
        #    cPickle.dump(nnet, f)

        if debug_on_test_only:
            continue

        # we check the validation loss on every epoch
        validation_losses = valid_scoref()
        this_validation_loss = numpy.mean(validation_losses)  # TODO this is a mean of means (with different lengths)
        print('  epoch %i, validation error %f' % \
              (epoch, this_validation_loss))
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            with open(output_file_name + '.pickle', 'wb') as f:
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
            print(('  epoch %i, test error of best model %f') %
                  (epoch, test_score))
        if patience <= iteration:  # TODO correct that
            done_looping = True
            break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f, '
           'with test performance %f') %
                 (best_validation_loss, test_score))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    with open(output_file_name + '_final.pickle', 'wb') as f:
        cPickle.dump(nnet, f)

if __name__=='__main__':
    arguments = docopt.docopt(__doc__, version='run_exp version 0.1')
    dataset_path=DEFAULT_DATASET
    if arguments['--dataset-path'] != None:
        dataset_path = arguments['--dataset-path']
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
    debug_print = 0
    if arguments['--debug-print']:
        debug_print = int(arguments['--debug-print'])
    debug_time = False
    if arguments['--debug-time']:
        debug_time = True
    debug_plot = 0
    if arguments['--debug-plot']:
        debug_plot = int(arguments['--debug-plot'])

    run(dataset_path=dataset_path, dataset_name=dataset_name,
        iterator_type=iterator_type, batch_size=batch_size,
        nframes=nframes, features=features,
        init_lr=init_lr, max_epochs=max_epochs, 
        network_type=network_type, trainer_type=trainer_type,
        #layers_types=[ReLU, ReLU, ReLU, ReLU, ReLU],
        #layers_sizes=[1000, 1000, 1000, 1000],  # TODO in opts
        layers_types=[ReLU, ReLU],
        layers_sizes=[200],  # TODO in opts
        recurrent_connections=[],  # TODO in opts
        prefix_fname=prefix_fname,
        debug_on_test_only=debug_on_test_only,
        debug_print=debug_print,
        debug_time=debug_time,
        debug_plot=debug_plot)
