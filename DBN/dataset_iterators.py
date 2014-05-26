MIN_FRAMES_PER_SENTENCE = 26
import numpy
from collections import defaultdict

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
        ret = numpy.zeros((x.shape[0], x.shape[1] * nf), dtype='float32')
        ba = (nf - 1) / 2  # before/after
        for i in xrange(x.shape[0]):
            ret[i] = numpy.pad(x[max(0, i - ba):i + ba +1].flatten(),
                    (max(0, (ba -i) * x.shape[1]),
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
