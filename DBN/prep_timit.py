import theano, copy, sys
import theano.tensor as T
import cPickle
import numpy as np

N_FRAMES = 11 # 5 before and 5 after the labeled frame
BORROW = True

def prep_data(dataset, unit=True, nframes=1):
    try:
        train_x = np.load(dataset + "/aligned_train_xdata.npy")
        train_y = np.load(dataset + "/aligned_train_ylabels.npy")
        test_x = np.load(dataset + "/aligned_test_xdata.npy")
        test_y = np.load(dataset + "/aligned_test_ylabels.npy")

    except:
        print >> sys.stderr, "you nee the .npy python arrays"
        print >> sys.stderr, dataset + "/aligned_train_xdata.npy"
        print >> sys.stderr, dataset + "/aligned_train_ylabels.npy"
        print >> sys.stderr, dataset + "/aligned_test_xdata.npy"
        print >> sys.stderr, dataset + "/aligned_test_ylabels.npy"
        sys.exit(-1)

    ### Feature values (Xs)
    if unit:
        train_x -= train_x.min() # TODO or do that line by line???
        train_x /= train_x.max()
        test_x -= test_x.min()
        test_x /= test_x.max()

    train_x_f = np.zeros((train_x.shape[0], nframes * train_x.shape[1]), dtype='float32')
    ba = (nframes - 1) / 2 # before / after
    for i in xrange(train_x.shape[0]):
        train_x_f[i] = np.pad(train_x[max(0, i - ba):i + ba + 1].flatten(),
                (max(0, (ba - i) * train_x.shape[1]), 
                    max(0, ((i+ba+1) - train_x.shape[0]) * train_x.shape[1])),
                'constant', constant_values=(0,0))

    test_x_f = np.zeros((test_x.shape[0], nframes * test_x.shape[1]), dtype='float32')
    for i in xrange(test_x.shape[0]):
        test_x_f[i] = np.pad(test_x[max(0, i - ba):i + ba + 1].flatten(),
                (max(0, (ba - i) * test_x.shape[1]), 
                    max(0, ((i+ba+1) - test_x.shape[0]) * test_x.shape[1])),
                'constant', constant_values=(0,0))

    ### Labels (Ys)
    from collections import Counter
    c = Counter(train_y)
    to_int = dict([(k, c.keys().index(k)) for k in c.iterkeys()])
    to_state = dict([(c.keys().index(k), k) for k in c.iterkeys()])
    cPickle.dump('to_int_and_to_state_dicts_tuple.pickle', (to_int, to_state))

    train_y_f = np.zeros((train_y.shape[1], 1), dtype='int32')
    for i, e in enumerate(train_y):
        train_y_f[i] = to_int[e]

    test_y_f = np.zeros((test_y.shape[1], 1), dtype='int32')
    for i, e in enumerate(test_y):
        test_y_f[i] = to_int[e]

    return [train_x_f, train_y_f, test_x_f, test_y_f]

def load_data(dataset, nframes=N_FRAMES):
    [train_x, train_y, test_x, test_y] = prep_data(dataset, 
            unit=True, nframes=nframes)
    
    train_set_x = theano.shared(train_x, borrow=BORROW)
    train_set_y = theano.shared(np.asarray(train_y, dtype=theano.config.floatX),
            borrow=BORROW)
    train_set_y = T.cast(train_set_y, 'int32')
    test_set_x = theano.shared(test_x, borrow=BORROW)
    test_set_y = theano.shared(np.asarray(test_y, dtype=theano.config.floatX), borrow=BORROW)
    test_set_y = T.cast(test_set_y, 'int32')
    return [(train_set_x, train_set_y), 
            (copy.deepcopy(test_set_x), copy.deepcopy(test_set_y)),
            (test_set_x, test_set_y)] # TODO validation dataset != training
