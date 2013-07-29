import pylab as pl
import numpy as np
import sys, cPickle

if len(sys.argv) < 3:
    print "python plot_ema.py ema_file.npy columns_dict.pickle"

with open(sys.argv[1]) as f:
    data = np.load(f)
with open(sys.argv[2]) as f:
    columns = cPickle.load(f)


n_dim = data.shape[1] - 2
n_plots = n_dim #* 3

x_y_n = int(np.ceil(np.sqrt(n_plots)))
for i in xrange(n_dim):
    pl.subplot(x_y_n, x_y_n -1, i+1)
    pl.plot(data[:, i+2])
    tmp_diff = np.pad(np.diff(data[:, i+2]),
            (1, 0),
            'constant', constant_values=(0.0, 0.0))
    pl.plot(tmp_diff)
    tmp_accel = np.pad(np.diff(tmp_diff),
            (1, 0),
            'constant', constant_values=(0.0, 0.0))
    pl.plot(tmp_accel)
    pl.ylabel(columns[i+2])
pl.subplots_adjust(wspace=0.5, hspace=0.5)
pl.legend(['pos', 'speed', 'accel'])
pl.suptitle('EMA for ' + sys.argv[1])
#pl.show()
pl.savefig('test_plot.png')

