import theano, copy, sys
import theano.tensor as T
import cPickle
import numpy as np

BORROW = True # True makes it faster with the GPU
USE_CACHING = True # beware if you use RBM / GRBM alternatively, set it to False
TRAIN_CLASSIFIERS = True # train sklearn classifiers to compare the DBN to

def padding(nframes, x, y):
    # dirty hacky padding
    ba = (nframes - 1) / 2 # before // after
    x2 = copy.deepcopy(x)
    on_x2 = False
    x_f = np.zeros((x.shape[0], nframes * x.shape[1]), dtype='float32')
    for i in xrange(x.shape[0]):
        if y[i] == '!ENTER[2]' and y[i-1] != '!ENTER[2]': # TODO general case
            on_x2 = not on_x2
            if on_x2:
                x2[i - ba:i,:] = 0.0
            else:
                x[i - ba:i,:] = 0.0
        if i+ba < y.shape[0] and '!EXIT' in y[i] and not '!EXIT' in y[i+ba]: # TODO general
            if on_x2:
                x2[i+ba:i+2*ba+1,:] = 0.0
            else:
                x[i+ba:i+2*ba+1,:] = 0.0
        if on_x2:
            x_f[i] = np.pad(x2[max(0, i - ba):i + ba + 1].flatten(),
                    (max(0, (ba - i) * x.shape[1]), 
                        max(0, ((i+ba+1) - x.shape[0]) * x.shape[1])),
                    'constant', constant_values=(0,0))
        else:
            x_f[i] = np.pad(x[max(0, i - ba):i + ba + 1].flatten(),
                    (max(0, (ba - i) * x.shape[1]), 
                        max(0, ((i+ba+1) - x.shape[0]) * x.shape[1])),
                    'constant', constant_values=(0,0))
    return x_f


def prep_data(dataset, nframes=1, unit=False, normalize=False, pca_whiten=False):
    try:
        train_x = np.load(dataset + "/aligned_train_xdata.npy")
        train_y = np.load(dataset + "/aligned_train_ylabels.npy")
        test_x = np.load(dataset + "/aligned_test_xdata.npy")
        test_y = np.load(dataset + "/aligned_test_ylabels.npy")

    except:
        print >> sys.stderr, "you need the .npy python arrays"
        print >> sys.stderr, "you can produce them with src/timit_to_numpy.py"
        print >> sys.stderr, "applied to the HTK force-aligned MLF train/test files"
        print >> sys.stderr, dataset + "/aligned_train_xdata.npy"
        print >> sys.stderr, dataset + "/aligned_train_ylabels.npy"
        print >> sys.stderr, dataset + "/aligned_test_xdata.npy"
        print >> sys.stderr, dataset + "/aligned_test_ylabels.npy"
        sys.exit(-1)

    print "train_x shape:", train_x.shape
    print "test_x shape:", test_x.shape

    if unit:
        ### Putting values on [0-1]
        train_x = (train_x - np.min(train_x, 0)) / np.max(train_x, 0)
        test_x = (test_x - np.min(test_x, 0)) / np.max(test_x, 0)
        # TODO or do that globally on all data
    if normalize:
        ### Normalizing (0 mean, 1 variance)
        # TODO or do that globally on all data
        train_x = (train_x - np.mean(train_x, 0)) / np.std(train_x, 0)
        test_x = (test_x - np.mean(test_x, 0)) / np.std(test_x, 0)
    if pca_whiten: # TODO change/correct that looking at prep_mocha_timit
        ### PCA whitening, beware it's sklearn's and thus stays in PCA space
        from sklearn.decomposition import PCA
        pca = PCA(n_components='mle', whiten=True)
        train_x = pca.fit_transform(train_x)
        test_x = pca.fit_transform(test_x)
    train_x_f = train_x
    test_x_f = test_x

    ### Feature values (Xs)
    print "preparing / padding Xs"
    if nframes > 1:
        train_x_f = padding(nframes, train_x, train_y)
        test_x_f = padding(nframes, test_x, test_y)

    ### Labels (Ys)
    from collections import Counter
    c = Counter(train_y)
    to_int = dict([(k, c.keys().index(k)) for k in c.iterkeys()])
    to_state = dict([(c.keys().index(k), k) for k in c.iterkeys()])
    with open('to_int_and_to_state_dicts_tuple.pickle', 'w') as f:
        cPickle.dump((to_int, to_state), f)

    print "preparing / int mapping Ys"
    train_y_f = np.zeros(train_y.shape[0], dtype='int32')
    for i, e in enumerate(train_y):
        train_y_f[i] = to_int[e]

    test_y_f = np.zeros(test_y.shape[0], dtype='int32')
    for i, e in enumerate(test_y):
        test_y_f[i] = to_int[e]

    if TRAIN_CLASSIFIERS:
        ### Training a SVM to compare results TODO
        #print "training a SVM" TODO

        ### Training a linear model (elasticnet) to compare results
        print("*** training a linear model with SGD ***")
        from sklearn import linear_model
        from sklearn.cross_validation import cross_val_score
        clf = linear_model.SGDClassifier(loss='modified_huber', penalty='elasticnet') # TODO change and CV params
        clf.fit(train_x, train_y_f)
        scores = cross_val_score(clf, test_x, test_y_f)
        print "score linear classifier (elasticnet, SGD trained)", scores.mean()
        with open('linear_elasticnet_classif.pickle', 'w') as f:
            cPickle.dump(clf, f)

        ### Training a random forest to compare results
        print("*** training a random forest ***")
        from sklearn.ensemble import RandomForestClassifier
        clf2 = RandomForestClassifier(n_jobs=-1, max_depth=None, min_samples_split=3) # TODO change and CV params
        clf2.fit(train_x, train_y_f)
        scores2 = cross_val_score(clf2, test_x, test_y_f)
        print "score random forest", scores2.mean()
        with open('random_forest_classif.pickle', 'w') as f:
            cPickle.dump(clf2, f)
        

        ### Feature selection
        print("*** feature selection now: ***")
        print(" - Feature importances for the random forest classifier")
        print clf2.feature_importances_
        from sklearn.feature_selection import SelectPercentile, f_classif 
        # SelectKBest TODO?
        selector = SelectPercentile(f_classif, percentile=10) # ANOVA
        selector.fit(train_x, train_y_f)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        print(" - ANOVA scoring (order of the MFCC)")
        print scores
        from sklearn.feature_selection import RFECV
        from sklearn.lda import LDA
        print(" - Recursive feature elimination with cross-validation with LDA")
        lda = LDA()
        #lda.fit(train_x, train_y_f, store_covariance=True)
        rfecv = RFECV(estimator=lda, step=1, scoring='accuracy')
        rfecv.fit(train_x, train_y_f)
        print("Optimal number of features : %d" % rfecv.n_features_)
        print("Ranking (order of the MFCC):")
        print rfecv.ranking_
        # TODO sample features combinations with LDA? kernels?

    return [train_x_f, train_y_f, test_x_f, test_y_f]


def load_data(dataset, nframes=11, unit=False, normalize=False, pca_whiten=False, cv_frac=0.2):
    def prep_and_serialize():
        [train_x, train_y, test_x, test_y] = prep_data(dataset, 
                nframes=nframes, unit=unit, normalize=normalize, pca_whiten=pca_whiten)
        with open('train_x_' + str(nframes) + '.npy', 'w') as f:
            np.save(f, train_x)
        with open('train_y_' + str(nframes) + '.npy', 'w') as f:
            np.save(f, train_y)
        with open('test_x_' + str(nframes) + '.npy', 'w') as f:
            np.save(f, test_x)
        with open('test_y_' + str(nframes) + '.npy', 'w') as f:
            np.save(f, test_y)
        print ">>> Serialized all train/test tables"
        return [train_x, train_y, test_x, test_y]

    if USE_CACHING:
        try: # try to load from serialized filed, bewa
            with open('train_x_' + str(nframes) + '.npy') as f:
                train_x = np.load(f)
            with open('train_y_' + str(nframes) + '.npy') as f:
                train_y = np.load(f)
            with open('test_x_' + str(nframes) + '.npy') as f:
                test_x = np.load(f)
            with open('test_y_' + str(nframes) + '.npy') as f:
                test_y = np.load(f)
        except: # do the whole preparation (normalization / padding)
            [train_x, train_y, test_x, test_y] = prep_and_serialize()
    else:
        [train_x, train_y, test_x, test_y] = prep_and_serialize()

    from sklearn import cross_validation
    X_train, X_validate, y_train, y_validate = cross_validation.train_test_split(train_x, train_y, test_size=cv_frac, random_state=0)
    
    train_set_x = theano.shared(X_train, borrow=BORROW)
    train_set_y = theano.shared(np.asarray(y_train, dtype=theano.config.floatX), borrow=BORROW)
    train_set_y = T.cast(train_set_y, 'int32')
    val_set_x = theano.shared(X_validate, borrow=BORROW)
    val_set_y = theano.shared(np.asarray(y_validate, dtype=theano.config.floatX), borrow=BORROW)
    val_set_y = T.cast(val_set_y, 'int32')
    test_set_x = theano.shared(test_x, borrow=BORROW)
    test_set_y = theano.shared(np.asarray(test_y, dtype=theano.config.floatX), borrow=BORROW)
    test_set_y = T.cast(test_set_y, 'int32')
    return [(train_set_x, train_set_y), 
            #(copy.deepcopy(test_set_x), copy.deepcopy(test_set_y)),
            (val_set_x, val_set_y),
            (test_set_x, test_set_y)] 
