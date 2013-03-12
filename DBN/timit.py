import numpy, theano, copy
import theano.tensor as T

DEBUG = 0
BORROW = True

def prep_data(dataset, unit=True):
    #[(<TensorType(float32, matrix)>, Elemwise{Cast{int32}}.0), (,), (,)]
    if DEBUG:
        print "============TRAIN SET=============="
    #x = numpy.load("/Users/gabrielsynnaeve/postdoc/htk_mfc/fcjf0_xdata.npy")
    x = numpy.load(dataset + "/train_xdata.npy")
    if unit: # TODO properly
        x -= x.min()
        x /= x.max()
    if DEBUG:
        print x
        print x.shape
        print train_set_x
    #y = numpy.load("/Users/gabrielsynnaeve/postdoc/htk_mfc/fcjf0_ylabels.npy")
    y = numpy.load(dataset + "/train_ylabels.npy")
    if DEBUG:
        print y
        print y.shape
    from collections import Counter
    to_int = Counter(y) # trick
    if DEBUG:
        print to_int
    for ind, (key, val) in enumerate(to_int.iteritems()):
        to_int[key] = ind
    if DEBUG:
        print to_int
        print "len(to_int)", len(to_int)
    # groupings
    to_int['sil'] = ind+9
    to_int['cl'] = to_int.get('sil')
    to_int['vcl'] = to_int.get('sil')
    to_int['epi'] = to_int.get('sil')
    to_int['l'] = to_int.get('el', ind+10)
    to_int['n'] = to_int.get('en', ind+11)
    to_int['zh'] = to_int.get('sh', ind+12)
    to_int['aa'] = to_int.get('ao', ind+13)
    to_int['ix'] = to_int.get('ih', ind+14)
    to_int['ax'] = to_int.get('ah', ind+15)
    to_int['ax-h'] = to_int.get('ax') # already grouped
    # folding
    to_int['ux'] = to_int.get('uw', ind+1)
    to_int['axr'] = to_int.get('er', ind+2)
    to_int['em'] = to_int.get('m', ind+3)
    to_int['nx'] = to_int.get('n', ind+4)
    to_int['eng'] = to_int.get('ng', ind+5)
    to_int['hv'] = to_int.get('hh', ind+6)
    to_int['pcl'] = to_int.get('cl', ind+7)
    to_int['tcl'] = to_int.get('cl', ind+7)
    to_int['kcl'] = to_int.get('cl', ind+7)
    to_int['qcl'] = to_int.get('cl', ind+7)
    to_int['bcl'] = to_int.get('vcl', ind+8)
    to_int['dcl'] = to_int.get('vcl', ind+8)
    to_int['gcl'] = to_int.get('vcl', ind+8)
    to_int['#h'] = to_int.get('sil')
    to_int['h#'] = to_int.get('sil')
    to_int['pau'] = to_int.get('sil')
    # TODO check that re-indexing:
    to_phones = {}
    for key, val in to_int.iteritems():
        to_phones[val] = to_phones.get(val, []) + [key]
    for ind, (key, val) in enumerate(to_phones.iteritems()):
        for phone in val:
            to_int[phone] = ind
    to_int = dict(to_int) # remove the Counter class
    print "to_phones:", to_phones
    print "to_int:", to_int
    #############

    if DEBUG:
        print to_int
        print "len(to_int)", len(to_int)
        print "len(set(to_int.values()))", len(set(to_int.values()))
    old_paper_phones = set(['iy', 'ih', 'eh', 'ae', 'ix', 'ax', 'ah', 'uw', 
        'uh', 'ao', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow', 'l', 'el', 'r', 'y',
        'w', 'er', 'm', 'n', 'en', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx', 'g',
        'p', 't', 'k', 'z', 'zh', 'v', 'f', 'th', 's', 'sh', 'hh', 'cl', 'vcl',
        'epi', 'sil']) # From "Speaker-Independant Phone Recognition Using HMM"
    assert(len(old_paper_phones) == 48)
    if DEBUG:
        print old_paper_phones - set(to_int.keys())
        print set(to_int.keys()) - old_paper_phones # TODO we seem to have the 'q' phone that is not in the 1989 paper "Speaker-Indep. Phone Recog. Using HMM"
        # thus we have 40 different phones instead of 39 /TODO
    yy = []
    for elem in y:
        yy.append(to_int[elem])
    #print train_set_y
    #print train_set_y
    if DEBUG:
        print "============/TRAIN SET=============="
        print "============TEST SET=============="
    #test_x = numpy.load("/Users/gabrielsynnaeve/postdoc/htk_mfc/fetb0_xdata.npy")
    test_x = numpy.load(dataset + "/test_xdata.npy")
    if unit: # TODO properly
        test_x -= test_x.min()
        test_x /= test_x.max()
    if DEBUG:
        print x
        print x.shape
    #test_y = numpy.load("/Users/gabrielsynnaeve/postdoc/htk_mfc/fetb0_ylabels.npy")
    test_y = numpy.load(dataset + "/test_ylabels.npy")
    test_yy = []
    for elem in test_y:
        test_yy.append(to_int[elem])
    #print test_set_y
    if DEBUG:
        print "============/TEST SET=============="
    return [x, yy, test_x, test_yy]


def load_data(dataset):
    [x, yy, test_x, test_yy] = prep_data(dataset)
    train_set_x = theano.shared(x, borrow=BORROW)
    train_set_y = theano.shared(numpy.asarray(yy, dtype=theano.config.floatX),
            borrow=BORROW)
    train_set_y = T.cast(train_set_y, 'int32')
    test_set_x = theano.shared(test_x, borrow=BORROW)
    test_set_y = theano.shared(numpy.asarray(test_yy, dtype=theano.config.floatX), borrow=BORROW)
    test_set_y = T.cast(test_set_y, 'int32')
    return [(train_set_x, train_set_y), 
            (copy.deepcopy(test_set_x), copy.deepcopy(test_set_y)),
            (test_set_x, test_set_y)] # TODO
