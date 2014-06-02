""" Takes a *.npy array as input and forms a *.npz file for ABX testing.
"""

import sys
import numpy as np

FRAMES_PER_SEC = 100  # features frames per second
FEATURES_RATE = 1. / FRAMES_PER_SEC

t = np.load(sys.argv[1])
tt = np.zeros(t.shape[0])
for i in xrange(tt.shape[0]):
    tt[i] = float(i)/FRAMES_PER_SEC + FEATURES_RATE / 2
np.savez(sys.argv[1].split('.')[0] + '.npz',
        features=t,
        time=tt)
