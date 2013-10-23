import theano, copy, sys, json, cPickle
import theano.tensor as T
import numpy as np

BORROW = True # True makes it faster with the GPU
USE_CACHING = True # beware if you use RBM / GRBM alternatively, set it to False
TRAIN_CLASSIFIERS_1_FRAME = False # train sklearn classifiers on 1 frame
TRAIN_CLASSIFIERS = False # train sklearn classifiers to compare the DBN to

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

def train_classifiers(train_x, train_y_f, test_x, test_y_f, articulatory=False, dataset_name='', classifiers=['lda'], nframes_mfcc=1):
    print("size of input layer (== dimension of the features space) %d" % train_x.shape[1])
    ### Training a SVM to compare results TODO
    #print "training a SVM" TODO
    if 'sgd' in classifiers:
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

    if 'rf' in classifiers:
        ### Training a random forest to compare results
        print("*** training a random forest ***")
        from sklearn.ensemble import RandomForestClassifier
        clf2 = RandomForestClassifier(n_jobs=-1, max_features='log2', min_samples_split=3) # TODO change and CV params
        clf2.fit(train_x, train_y_f)
        scores2 = cross_val_score(clf2, test_x, test_y_f)
        print "score random forest", scores2.mean()
        ###with open('random_forest_classif.pickle', 'w') as f: TODO TODO TODO
        ###    cPickle.dump(clf2, f)  TODO TODO TODO 

    if 'lda' in classifiers:
        print "*** training a linear discriminant classifier ***"
        from sklearn.lda import LDA
        from sklearn.metrics import confusion_matrix
        from sklearn import cross_validation

        def lda_on(train_x, train_y, test_x, test_y, feats_name='all_features'):
            lda = LDA()
            lda.fit(train_x, train_y, store_covariance=True)
            print feats_name, "(train):", lda.score(train_x, train_y)
            print feats_name, "(test):", lda.score(test_x, test_y)
            with open(dataset_name + '_lda_classif_' + feats_name + '.pickle', 'w') as f:
                cPickle.dump(lda, f)
            y_pred = lda.predict(test_x)
            X_train, X_validate, y_train, y_validate = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)
            lda.fit(X_train, y_train)
            print feats_name, "(validation):", lda.score(X_validate, y_validate)
            y_pred_valid = lda.predict(X_validate)
            cm_test = confusion_matrix(test_y, y_pred)
            cm_valid = confusion_matrix(y_validate, y_pred_valid)
            np.set_printoptions(threshold='nan')
            with open("cm_test" + feats_name + ".txt", 'w') as wf:
                print >> wf, cm_test
            with open("cm_valid" + feats_name + ".txt", 'w') as wf:
                print >> wf, cm_valid

        if articulatory:
            lda_on(train_x[:,:39*nframes_mfcc], train_y_f, test_x[:,:39*nframes_mfcc], test_y_f, feats_name='mfcc')
            lda_on(train_x[:,39*nframes_mfcc:], train_y_f, test_x[:,39*nframes_mfcc:], test_y_f, feats_name='arti')
        else:
            lda_on(train_x, train_y_f, test_x, test_y_f, feats_name='both')

    if 'featselec' in classifiers:
        ### Feature selection
        print("*** feature selection now: ***")
        print(" - Feature importances for the random forest classifier")
        print clf2.feature_importances_
        from sklearn.feature_selection import SelectPercentile, f_classif 
        # SelectKBest TODO?
        selector = SelectPercentile(f_classif, percentile=10) # ANOVA
        selector.fit(train_x, train_y_f)
        print selector.pvalues_
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        print(" - ANOVA scoring (order of the MFCC)")
        print scores
        from sklearn.feature_selection import RFECV
        print(" - Recursive feature elimination with cross-validation with LDA")
        lda = LDA()
        rfecv = RFECV(estimator=lda, step=1, scoring='accuracy')
        rfecv.fit(train_x, train_y_f)
        print("Optimal number of features : %d" % rfecv.n_features_)
        print("Ranking (order of the MFCC):")
        print rfecv.ranking_
        # TODO sample features combinations with LDA? kernels?


def prep_data(dataset, nframes=1, unit=False, normalize=False, student=False, 
              pca_whiten=0, gammatones=False, dataset_name=''):
    # TODO "standard split" as in /fhgfs/bootphon/scratch/gsynnaeve/TIMIT/std_split
    # normalize takes precedence over student's t-stat
    xname = "xdata"
    if gammatones:
        xname = "xgamma"
    try:
        train_x = np.load(dataset + "/aligned_train_" + xname + ".npy")
        train_y = np.load(dataset + "/aligned_train_ylabels.npy")
        test_x = np.load(dataset + "/aligned_test_" + xname + ".npy")
        test_y = np.load(dataset + "/aligned_test_ylabels.npy")

    except:
        print >> sys.stderr, "you need the .npy python arrays"
        print >> sys.stderr, "you can produce them with src/timit_to_numpy.py"
        print >> sys.stderr, "applied to the HTK force-aligned MLF train/test files"
        print >> sys.stderr, dataset + "/aligned_train_" + xname + ".npy"
        print >> sys.stderr, dataset + "/aligned_train_ylabels.npy"
        print >> sys.stderr, dataset + "/aligned_test_" + xname + ".npy"
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
    elif student: # mutually exclusive
        ### T-statistic
        train_x = (train_x - np.mean(train_x, 0)) / np.std(train_x, ddof=1)
        test_x = (test_x - np.mean(test_x, 0)) / np.std(test_x, 0, ddof=1)
    if pca_whiten: 
        ### PCA whitening, beware it's sklearn's and thus stays in PCA space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_whiten, whiten=True)
        if pca_whiten < 0:
            pca = PCA(n_components='mle', whiten=True)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)
        with open(dataset_name + '_pca_' + xname + '.pickle', 'w') as f:
            cPickle.dump(pca, f)
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
    with open(dataset_name + '_to_int_and_to_state_dicts_tuple.pickle', 'w') as f:
        cPickle.dump((to_int, to_state), f)

    print "preparing / int mapping Ys"
    train_y_f = np.zeros(train_y.shape[0], dtype='int32')
    for i, e in enumerate(train_y):
        train_y_f[i] = to_int[e]

    test_y_f = np.zeros(test_y.shape[0], dtype='int32')
    for i, e in enumerate(test_y):
        test_y_f[i] = to_int[e]

    if TRAIN_CLASSIFIERS_1_FRAME:
        train_classifiers(train_x, train_y_f, test_x, test_y_f, dataset_name=dataset_name) # ONLY 1 FRAME
    if TRAIN_CLASSIFIERS:
        train_classifiers(train_x_f, train_y_f, test_x_f, test_y_f, dataset_name=dataset_name, nframes_mfcc=nframes)

    return [train_x_f, train_y_f, test_x_f, test_y_f]


def load_data(dataset, nframes=11, unit=False, normalize=False, student=False, pca_whiten=0, cv_frac=0.2, gammatones=False, dataset_name='timit'):
    """ 
    params:
     - dataset: folder
     - nframes: number of frames to replicate/pad
     - unit?: put all the data into [0-1]
     - normalize?: (X-mean(X))/std(X)
     - student?: (X-mean(X))/std(X, deg_of_liberty=1)
     - pca_whiten: not if 0, MLE if < 0, number of components if > 0
     - cv_frac: cross validation fraction on the train set
     - gammatones?: use gammatones instead of MFCC
    """
    params = {'nframes_mfcc': nframes,
              'unit': unit,
              'normalize': normalize,
              'student': student,
              'pca_whiten_mfcc_path': 'pca_' + str(pca_whiten) + '.pickle' if pca_whiten else 0,
              'cv_frac': cv_frac,
              'theano_borrow?': BORROW,
              'use_caching?': USE_CACHING,
              'train_classifiers_1_frame?': TRAIN_CLASSIFIERS_1_FRAME,
              'train_classifiers?': TRAIN_CLASSIFIERS,
              'gammatones?': gammatones,
              'dataset_name': dataset_name}
    with open('prep_' + dataset_name + '_params.json', 'w') as f:
        f.write(json.dumps(params))

    gamma = "" # for filenames / caching
    if gammatones:
        gamma = "gamma_" # for filenames / caching

    def prep_and_serialize():
        [train_x, train_y, test_x, test_y] = prep_data(dataset, 
                nframes=nframes, unit=unit, normalize=normalize, 
                student=student, pca_whiten=pca_whiten, gammatones=gammatones,
                dataset_name=dataset_name)
        with open('train_x_' + dataset_name + '_' + gamma + str(nframes) + '.npy', 'w') as f:
            np.save(f, train_x)
        with open('train_y_' + dataset_name + '_' + gamma + str(nframes) + '.npy', 'w') as f:
            np.save(f, train_y)
        with open('test_x_' + dataset_name + '_' + gamma + str(nframes) + '.npy', 'w') as f:
            np.save(f, test_x)
        with open('test_y_' + dataset_name + '_' + gamma + str(nframes) + '.npy', 'w') as f:
            np.save(f, test_y)
        print ">>> Serialized all train/test tables"
        return [train_x, train_y, test_x, test_y]

    if USE_CACHING:
        try: # try to load from serialized filed, beware
            with open('train_x_' + dataset_name + '_' + gamma + str(nframes) + '.npy') as f:
                train_x = np.load(f)
            with open('train_y_' + dataset_name + '_' + gamma +  str(nframes) + '.npy') as f:
                train_y = np.load(f)
            with open('test_x_' + dataset_name + '_' + gamma +  str(nframes) + '.npy') as f:
                test_x = np.load(f)
            with open('test_y_' + dataset_name + '_' + gamma +  str(nframes) + '.npy') as f:
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
            (val_set_x, val_set_y),
            (test_set_x, test_set_y)] 
