MIN_FRAMES_PER_SENTENCE = 26
import numpy, theano
from collections import defaultdict
from itertools import izip
#from joblib import Memory
#mem = Memory(cachedir='/fhgfs/bootphon/scratch/gsynnaeve/tmp_npy', verbose=0)

# TODO benchmark shared instead of numpy arrays.


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

    def __init__(self, x1, x2, y, nframes=1, batch_size=1, margin=False):
        # x1 and x2 are tuples or arrays that are [nframes, nfeatures]
        self._x1 = x1
        self._x2 = x2
        self._y = [numpy.zeros(x.shape[0], dtype='int8') for x in self._x1]
        # self._y says if frames in x1 and x2 are same (1) or different (0)
        for ii, yy in enumerate(y):
            self._y[ii][:] = yy
        self._nframes = nframes
        self._memoized_x = defaultdict(lambda: {})
        self._margin = 0
        self._nwords = batch_size
        if margin:
            # margin says if we pad taking a margin into account
            self._margin = (self._nframes - 1) / 2
        self._x1_mem = []
        self._x2_mem = []
        self._y_mem = []


    def _memoize(self, i):
        """ Computes the corresponding x1/x2/y for the given i depending on the
        self._nframes (stacking x1/x2 features for self._nframes), and
        self._nwords (number of words per mini-batch).
        """
        ii = i/self._nwords
        if ii < len(self._x1_mem) and ii < len(self._x2_mem):
            return [[self._x1_mem[ii], self._x2_mem[ii]], self._y_mem[ii]]

        nf = self._nframes
        def local_pad(x):
            if nf <= 1:
                return x
            if self._margin:
                ma = self._margin
                ba = (nf - 1) / 2  # before/after
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
        ii = i/self._nwords
        return [[self._x1_mem[ii], self._x2_mem[ii]], self._y_mem[ii]]

    def __iter__(self):
        for i in xrange(0, len(self._y), self._nwords):
            yield self._memoize(i)
