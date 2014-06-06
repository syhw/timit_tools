from nnet_archs import ABNeuralNet, NeuralNet
from classifiers import LogisticRegression
import sys, cPickle
from prep_timit import load_data
import numpy as np
import json

usage = "python evaluate_abnnets.py abnnet.pickle"
DATASET = '/fhgfs/bootphon/scratch/gsynnaeve/TIMIT/train_dev_test_split'
DEBUG = False  # just works on valid/test sets
NFEATURES_PER_FRAME = 40
NFRAMES = 7
REMOVE_ENTER_EXIT = True
FOLDINGS = True

if not len(sys.argv):
    print usage
    sys.exit(-1)

abnnet = None
with open(sys.argv[1], 'rb') as f:
    abnnet = cPickle.load(f)
if abnnet is None:
    print "couldnt load the abnnet"
    sys.exit(-1)

print abnnet
NFRAMES = abnnet.layers_ins[0] / NFEATURES_PER_FRAME
print abnnet.layers[-1].output
print abnnet.layers[-2].output


def createLogisticRegression(n_ins, n_outs):
    numpy_rng = np.random.RandomState(123)
    ret = NeuralNet(numpy_rng=numpy_rng,
            n_ins=n_ins,
            layers_types=[LogisticRegression],
            layers_sizes=[],
            n_outs=n_outs,
            debugprint=False)
    return ret



data = load_data(DATASET, nframes=NFRAMES, features='fbank', scaling='none',
        cv_frac='fixed', speakers=False, numpy_array_only=True)
d = np.load("mean_std.npz")
mean = d['mean']
std = d['std']
mean = np.tile(mean, NFRAMES)
std = np.tile(std, NFRAMES)

train_set_x, train_set_y = data[0]
valid_set_x, valid_set_y = data[1]
test_set_x, test_set_y = data[2]
train_set_x = np.asarray((train_set_x - mean) / std, dtype='float32')
valid_set_x = np.asarray((valid_set_x - mean) / std, dtype='float32')
test_set_x = np.asarray((test_set_x - mean) / std, dtype='float32')

if REMOVE_ENTER_EXIT:
    to_int = {}
    to_state = {}
    with open('timit_to_int_and_to_state_dicts_tuple.pickle') as f:
        to_int, to_state = cPickle.load(f)
    phones_to_remove = [v for k, v in to_int.iteritems() if "ENTER" in k or "EXIT" in k]
    tmp = np.in1d(train_set_y, phones_to_remove) == False
    train_set_x = train_set_x[tmp, :]
    train_set_y = train_set_y[tmp]
    tmp = np.in1d(valid_set_y, phones_to_remove) == False
    valid_set_x = valid_set_x[tmp, :]
    valid_set_y = valid_set_y[tmp]
    tmp = np.in1d(test_set_y, phones_to_remove) == False
    test_set_x = test_set_x[tmp, :]
    test_set_y = test_set_y[tmp]

if FOLDINGS:
    try:
        tmp = np.load("folded_ys.npz")
        train_set_y = tmp['train']
        valid_set_y = tmp['valid']
        test_set_y = tmp['test']
    except IOError:
        with open("../timit_foldings.json") as rf:
            foldings = json.load(rf)
        phones_to_fold = [(v, foldings[k.split('[')[0]])
                for k, v in to_int.iteritems() if k.split('[')[0] in foldings]
        ptf = dict(phones_to_fold)
        tmp = np.ndarray((train_set_y.shape[0],), dtype='|S10')
        for i in xrange(train_set_y.shape[0]):
            if train_set_y[i] in phones_to_fold:
                tmp[i] = ptf[train_set_y[i]]
            else:
                tmp[i] = to_state[train_set_y[i]].split('[')[0] 
        train_set_y = tmp
        tmp = np.ndarray((valid_set_y.shape[0],), dtype='|S10')
        for i in xrange(valid_set_y.shape[0]):
            if valid_set_y[i] in phones_to_fold:
                tmp[i] = ptf[valid_set_y[i]]
            else:
                tmp[i] = to_state[valid_set_y[i]].split('[')[0] 
        valid_set_y = tmp
        tmp = np.ndarray((test_set_y.shape[0],), dtype='|S10')
        for i in xrange(test_set_y.shape[0]):
            if test_set_y[i] in phones_to_fold:
                tmp[i] = ptf[test_set_y[i]]
            else:
                tmp[i] = to_state[test_set_y[i]].split('[')[0] 
        test_set_y = tmp
        np.savez("folded_ys.npz", train=train_set_y,
                valid=valid_set_y, test=test_set_y)

if DEBUG:
    print "training on the validation set"
    train_set_x, train_set_y = valid_set_x, valid_set_y

print "training set:",
print train_set_x.shape
print "test set:",
print test_set_x.shape

transform = abnnet.transform_x1()
NPIECES = 10
transformed_train_set_x = np.concatenate([transform(train_set_x[i*train_set_x.shape[0]/NPIECES:
    (i+1)*train_set_x.shape[0]/NPIECES]) for i in xrange(NPIECES)])
transformed_test_set_x = transform(test_set_x)

#tset_mean = transformed_test_set_x.mean(axis=0)
#tset_std = transformed_test_set_x.std(axis=0)
#tset_mean = transformed_test_set_x.mean()
#tset_std = transformed_test_set_x.std()
#transformed_train_set_x = (transformed_train_set_x - tset_mean) / tset_std
#transformed_test_set_x = (transformed_test_set_x - tset_mean) / tset_std
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
transformed_train_set_x = mms.fit_transform(transformed_train_set_x)
transformed_test_set_x = mms.transform(transformed_test_set_x)

print "training embedded set:",
print transformed_train_set_x.shape
print "test embedded set:",
print transformed_test_set_x.shape

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from lightning.classification import CDClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_set_y = np.asarray(le.fit_transform(train_set_y), dtype='int32')
valid_set_y = np.asarray(le.transform(valid_set_y), dtype='int32')
test_set_y = np.asarray(le.transform(test_set_y), dtype='int32')

clfs = [#SGDClassifier(loss='hinge', penalty='l2'),
        "LogisticRegression",
        CDClassifier(penalty="l1/l2",
            loss="squared_hinge",
            multiclass=True,
            max_iter=20,
            alpha=1e-4,
            C=1.0 / train_set_x.shape[0],
            tol=1e-3),
        CDClassifier(penalty="l1/l2",
            loss="log",
            multiclass=True,
            max_iter=20,
            alpha=1e-4,
            C=1.0 / train_set_x.shape[0],
            tol=1e-3),
        #svm.LinearSVC(), 
        svm.SVC(kernel='rbf', cache_size=8000, max_iter=20)]
for clf in clfs:
    print clf
    clf2 = None
    if clf == "LogisticRegression":
        y_size = len(set(train_set_y))
        clf = createLogisticRegression(train_set_x.shape[1], y_size)
        clf2 = createLogisticRegression(transformed_train_set_x.shape[1], y_size)
    y_pred = None
    if hasattr(clf, 'fit_transform'):
        y_pred = clf.fit_transform(train_set_x, train_set_y)
    else:
        clf.fit(train_set_x, train_set_y)
    print "Training accuracy on fbanks:", 
    if y_pred is None:
        y_pred = clf.predict(train_set_x)
    print accuracy_score(train_set_y, y_pred)
    print "Test accuracy on fbanks:", 
    print accuracy_score(test_set_y, clf.predict(test_set_x))

    if clf2 is not None:
        clf = clf2
    y_pred = None
    if hasattr(clf, 'fit_transform'):
        y_pred = clf.fit_transform(transformed_train_set_x, train_set_y)
    else:
        clf.fit(transformed_train_set_x, train_set_y)
    print "Training accuracy on embedding:", 
    if y_pred is None:
        y_pred = clf.predict(transformed_train_set_x)
    print accuracy_score(train_set_y, y_pred)
    print "Test accuracy on embedding:", 
    print accuracy_score(test_set_y, clf.predict(transformed_test_set_x))


