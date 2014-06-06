MIN_FRAMES_PER_SENTENCE = 26
import numpy, theano
from collections import defaultdict
import random, joblib
from multiprocessing import cpu_count

# TODO benchmark shared instead of numpy arrays.

def pad(x, nf, ma=0):
    """ pad x for nf frames with margin ma. """
    ba = (nf - 1) / 2  # before/after
    if ma:
        ret = numpy.zeros((x.shape[0] - 2 * ma, x.shape[1] * nf),
                dtype=theano.config.floatX)
        if ba <= ma:
            for j in xrange(ret.shape[0]):
                ret[j] = x[j:j + 2*ma + 1].flatten()
        else:
            for j in xrange(ret.shape[0]):
                ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                        (max(0, (ba - j) * x.shape[1]),
                            max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                        'constant', constant_values=(0, 0))
        return ret
    else:
        ret = numpy.zeros((x.shape[0], x.shape[1] * nf),
                dtype=theano.config.floatX)
        for j in xrange(x.shape[0]):
            ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                    (max(0, (ba - j) * x.shape[1]),
                        max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                    'constant', constant_values=(0, 0))
        return ret


from dtw import DTW
def do_dtw(x1, x2):
    dtw = DTW(x1, x2, return_alignment=1)
    return dtw[0], dtw[-1][1], dtw[-1][2]


class DatasetSentencesIterator(object):
    """ An iterator on sentences of the dataset. """

    def __init__(self, x, y, phn_to_st, nframes=1, batch_size=None):
        # batch_size is ignored
        self._x = x
        self._y = numpy.asarray(y)
        self._start_end = [[0]]
        self._nframes = nframes
        self._memoized_x = defaultdict(lambda: {})
        i = 0
        for i, s in enumerate(self._y == phn_to_st['!ENTER[2]']):
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
        ret = numpy.zeros((x.shape[0], x.shape[1] * nf),
                dtype=theano.config.floatX)
        ba = (nf - 1) / 2  # before/after
        for i in xrange(x.shape[0]):
            ret[i] = numpy.pad(x[max(0, i - ba):i + ba +1].flatten(),
                    (max(0, (ba - i) * x.shape[1]),
                        max(0, ((i + ba + 1) - x.shape[0]) * x.shape[1])),
                    'constant', constant_values=(0, 0))
        self._memoized_x[start][end] = ret
        return ret

    def __iter__(self):
        for start, end in self._start_end:
            if self._nframes > 1:
                yield self._stackpad(start, end), self._y[start:end]
            else:
                yield self._x[start:end], self._y[start:end]


class DatasetSentencesIteratorPhnSpkr(DatasetSentencesIterator):
    """ An iterator on sentences of the dataset, specialized for datasets
    with both phones and speakers in y labels. """
    def __init__(self, x, y, phn_to_st, nframes=1, batch_size=None):
        super(DatasetSentencesIteratorPhnSpkr, self).__init__(x, y[0], phn_to_st, nframes, batch_size)
        self._y_spkr = numpy.asarray(y[1])

    def __iter__(self):
        for start, end in self._start_end:
            if self._nframes > 1:
                yield self._stackpad(start, end), self._y[start:end], self._y_spkr[start:end]
            else:
                yield self._x[start:end], self._y[start:end], self._y_spkr[start:end]


class DatasetBatchIterator(object):
    def __init__(self, x, y, phn_to_st, nframes=1, batch_size=None):
        pass
        # TODO


class DatasetDTWIterator(object):
    """ An iterator on dynamic time warped words of the dataset. """

    def __init__(self, x1, x2, y, nframes=1, batch_size=1, marginf=0):
        # x1 and x2 are tuples or arrays that are [nframes, nfeatures]
        self._x1 = x1
        self._x2 = x2
        self._y = [numpy.zeros(x.shape[0], dtype='int8') for x in self._x1]
        # self._y says if frames in x1 and x2 are same (1) or different (0)
        for ii, yy in enumerate(y):
            self._y[ii][:] = yy
        self._nframes = nframes
        self._memoized_x = defaultdict(lambda: {})
        self._nwords = batch_size
        self._margin = marginf
        # marginf says if we pad taking a number of frames as margin
        self._x1_mem = []
        self._x2_mem = []
        self._y_mem = []


    def _memoize(self, i):
        """ Computes the corresponding x1/x2/y for the given i depending on the
        self._nframes (stacking x1/x2 features for self._nframes), and
        self._nwords (number of words per mini-batch).
        """
        ind = i/self._nwords
        if ind < len(self._x1_mem) and ind < len(self._x2_mem):
            return [[self._x1_mem[ind], self._x2_mem[ind]], self._y_mem[ind]]

        nf = self._nframes
        def local_pad(x):  # TODO replace with pad global function
            if nf <= 1:
                return x
            if self._margin:
                ma = self._margin
                ba = (nf - 1) / 2  # before/after
                if x.shape[0] - 2*ma <= 0:
                    print "shape[0]:", x.shape[0]
                    print "ma:", ma
                if x.shape[1] * nf <= 0:
                    print "shape[1]:", x.shape[1]
                    print "nf:", nf
                ret = numpy.zeros((x.shape[0] - 2 * ma, x.shape[1] * nf),
                        dtype=theano.config.floatX)
                if ba <= ma:
                    for j in xrange(ret.shape[0]):
                        ret[j] = x[j:j + 2*ma + 1].flatten()
                else:
                    for j in xrange(ret.shape[0]):
                        ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                                (max(0, (ba - j) * x.shape[1]),
                                    max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                                'constant', constant_values=(0, 0))
                return ret
            else:
                ret = numpy.zeros((x.shape[0], x.shape[1] * nf),
                        dtype=theano.config.floatX)
                ba = (nf - 1) / 2  # before/after
                for j in xrange(x.shape[0]):
                    ret[j] = numpy.pad(x[max(0, j - ba):j + ba +1].flatten(),
                            (max(0, (ba - j) * x.shape[1]),
                                max(0, ((j + ba + 1) - x.shape[0]) * x.shape[1])),
                            'constant', constant_values=(0, 0))
                return ret
        
        def cut_y(y):
            ma = self._margin
            if nf <= 1 or ma == 0:
                return numpy.asarray(y, dtype='int8')
            ret = numpy.zeros((y.shape[0] - 2 * ma), dtype='int8')
            for j in xrange(ret.shape[0]):
                ret[j] = y[j+ma]
            return ret

        x1_padded = [local_pad(self._x1[i+k]) for k 
                in xrange(self._nwords) if i+k < len(self._x1)]
        x2_padded = [local_pad(self._x2[i+k]) for k
                in xrange(self._nwords) if i+k < len(self._x2)]
        assert x1_padded[0].shape[0] == x2_padded[0].shape[0]
        y_padded = [cut_y(self._y[i+k]) for k in
            xrange(self._nwords) if i+k < len(self._y)]
        assert x1_padded[0].shape[0] == len(y_padded[0])
        self._x1_mem.append(numpy.concatenate(x1_padded))
        self._x2_mem.append(numpy.concatenate(x2_padded))
        self._y_mem.append(numpy.concatenate(y_padded))
        return [[self._x1_mem[ind], self._x2_mem[ind]], self._y_mem[ind]]

    def __iter__(self):
        for i in xrange(0, len(self._y), self._nwords):
            yield self._memoize(i)


class DatasetDTReWIterator(DatasetDTWIterator):
    """ TODO """

    def __init__(self, data_same, mean, std, nframes=1, batch_size=1, marginf=0, only_same=False):
        dtw_costs = zip(*data_same)[5]
        self._orig_x1s = zip(*data_same)[3]
        self._orig_x2s = zip(*data_same)[4]
        self._words_frames = numpy.asarray([fb.shape[0] for fb in self._orig_x1s])
        self.print_mean_DTW_costs(dtw_costs)

        self._mean = mean
        self._std = std
        self._nframes = nframes
        self._nwords = batch_size
        self._margin = marginf
        self._only_same = only_same
        # marginf says if we pad taking a number of frames as margin

        same_spkr = 0
        for i, tup in enumerate(data_same):
            if tup[1] == tup[2]:
                same_spkr += 1
        ratio = same_spkr * 1. / len(data_same)
        print "ratio same spkr / all for same:", ratio
        data_diff = []
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
            p1 = data_same[word_1][3+wt1]
            p2 = data_same[word_2][3+wt2]
            r1 = p1[:min(len(p1), len(p2))]
            r2 = p2[:min(len(p1), len(p2))]
            data_diff.append((r1, r2))
        ratio = same_spkr_diff * 1. / len(data_diff)
        print "ratio same spkr / all for diff:", ratio

        self._data_same = zip(zip(*data_same)[3], zip(*data_same)[4],
                zip(*data_same)[-2], zip(*data_same)[-1])
        self._data_diff = data_diff

        self.remix()

        if self._nframes > 1:
            # pad the orig_xes1/2 once and for all
            self._orig_x1s = joblib.Parallel(n_jobs=cpu_count()-3)(
                    joblib.delayed(pad)(x, self._nframes, self._margin)
                    for x in self._orig_x1s)
            self._orig_x2s = joblib.Parallel(n_jobs=cpu_count()-3)(
                    joblib.delayed(pad)(x, self._nframes, self._margin)
                    for x in self._orig_x2s)


    def remix(self):
        x_same = [((e[0][e[-2]] - self._mean) / self._std, (e[1][e[-1]] - self._mean) / self._std)
                for e in self._data_same]
        y_same = [[1 for _ in xrange(len(e[0]))] for i, e in enumerate(x_same)]
        if not self._only_same:
            x_diff = [((e[0] - self._mean) / self._std, (e[1] - self._mean) / self._std)
                    for e in self._data_diff]
            random.shuffle(x_diff)
            y_diff = [[0 for _ in xrange(len(e[0]))] for i, e in enumerate(x_diff)]
            y = [j for i in zip(y_same, y_diff) for j in i]
            x = [j for i in zip(x_same, x_diff) for j in i]
        else:
            x = x_same
            y = y_same
        x1, x2 = zip(*x)
        # x1 and x2 are tuples or arrays that are [nframes, nfeatures]
        self._x1 = x1
        self._x2 = x2
        self._y = [numpy.zeros(x.shape[0], dtype='int8') for x in self._x1]
        # self._y says if frames in x1 and x2 are same (1) or different (0)
        for ii, yy in enumerate(y):
            self._y[ii][:] = yy
        self._memoized_x = defaultdict(lambda: {})
        self._x1_mem = []
        self._x2_mem = []
        self._y_mem = []

    def recompute_DTW(self, transform_f):
        from itertools import izip
        xes1 = map(transform_f, self._orig_x1s)
        xes2 = map(transform_f, self._orig_x2s)
        res = joblib.Parallel(n_jobs=cpu_count()-3)(joblib.delayed(do_dtw)
                (x1, x2) for x1, x2 in izip(xes1, xes2))
        dtw_costs = zip(*res)[0]
        self.print_mean_DTW_costs(dtw_costs)
        ds = zip(*self._data_same)
        rs = zip(*res)
        data_same_00shapes = self._data_same[0][0].shape
        data_same_01shapes = self._data_same[0][1].shape
        print data_same_00shapes
        print data_same_01shapes
        self._data_same = zip(ds[0], ds[1], rs[-2], rs[-1])
        data_same_00shapes = self._data_same[0][0].shape
        data_same_01shapes = self._data_same[0][1].shape
        print data_same_00shapes
        print data_same_01shapes
        self._margin = 0
        self.remix()


    def print_mean_DTW_costs(self, dtw_costs):
        print "mean DTW cost", numpy.mean(dtw_costs), "std dev", numpy.std(dtw_costs)
        print "mean word length in frames", numpy.mean(self._words_frames), "std dev", numpy.std(self._words_frames)
        print "mean DTW cost per frame", numpy.mean(dtw_costs/self._words_frames), "std dev", numpy.std(dtw_costs/self._words_frames)
