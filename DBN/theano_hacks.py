import time
import gc
import numpy as np


def theano_memory_hack(func_exp, local_vars, 
                       input_exps=('input',),
                       msize_best=None,
                       msize_start=1024, msize_factor=2,
                       verbose=False):
    """Super-badass hacky way of computing theano expression on large datasets.

    Just the first dimension will be "sliced". This is especially useful
    to get around memory errors while using the GPU.

    XXX: report this annoyance to theano-dev

    Variable used (don't use them in your expressions):
    * slice_vars[i]
    * input_vars[i]
    In general be careful about the expressions you use...

    XXX: use dictionaries / templates
    """

    assert len(input_exps) > 0

    errors_handled = [
        'Was not able to allocate output!',
        'expected a CudaNdarray, not None',
        'Could not allocate memory on device',
        'Error allocating',
    ]

    locals().update(local_vars)

    n_elements = len(eval(input_exps[0]))
    input_vars = [eval(input_exp) for input_exp in input_exps]
    for input_var in input_vars:
        assert len(input_var) == n_elements

    if msize_best is None:
        msize = msize_start
        msize_best = msize
        grow_msize = True
    else:
        msize = msize_best
        grow_msize = False

    output = None
    i = 0
    while i < n_elements:
        b = time.time()
        if verbose:
            print msize, msize_best

        try:
            slice_vars = [input_var[i:i+msize]
                          for input_var in input_vars]
            slice_output = eval(func_exp)
            msize_best = msize  # it worked with msize
        except Exception, err:
            gc.collect()
            if verbose:
                print err.message
            # hacky way to detect a out of memory error in theano
            done = False
            while not done:
                try:
                    if np.sum([err.message in error for error in errors_handled]) > 0:
                        if verbose:
                            print "!!! Memory error detected: hacking around..."
                        slice_vars = [input_var[i:i+msize_best]
                                      for input_var in input_vars]
                        slice_output = eval(func_exp)
                        grow_msize = False
                        msize = msize_best
                        done = True
                except Exception, err:
                    msize_best /= msize_factor
                    msize = msize_best

        if output is None:
            output = [slice_output]
        else:
            output.append(slice_output)

        i += msize

        if verbose:
            print 't: %.3f, msize_best: %d' % (time.time() - b, msize_best)

        if grow_msize:
            msize *= msize_factor

    #assert len(output) == n_elements, (len(output), n_elements)
    #assert output.dtype == input_vars[0].dtype

    return output, msize_best
