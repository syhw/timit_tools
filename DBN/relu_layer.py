import numpy
import theano
from theano import tensor as T
from theano import shared

# TODO maybe put adagrad/adadelta parameters in these classes
# TODO denoising ReLU auto-encoder
# TODO fast dropout using Wang & Manning 2013
#self.mask = srng.normal(avg=T.mean(self.output), std=T.std(self.output), size=self.output.shape) CORRECT THAT


def relu_f(v):
    """ Wrapper to quickly change the rectified linear unit function """
    # could do: T.switch(v > 0., v, 0 * v), quick benchmark is:
    # In [ ]: x = shared(np.asarray(np.random.random((1000, 1000)) ,dtype='float32'))
    # In [ ]: def relu_abs(v):
    #         return (v + abs(v))/2.
    # In [ ]: %timeit relu_abs(x)
    # 100 loops, best of 3: 9.26 ms per loop
    # In [ ]: def relu_switch(v):
    #         return T.switch(v>0., v, 0*v)
    # In [ ]: %timeit relu_switch(x)
    # 100 loops, best of 3: 11.7 ms per loop
    # In [ ]: %timeit T.grad(T.sum(relu_switch(x)), x)
    # 10 loops, best of 3: 86.5 ms per loop
    # In [ ]: %timeit T.grad(T.sum(relu_abs(x)), x)
    # 10 loops, best of 3: 71.8 ms per loop
    return (v + abs(v)) / 2.
    #return T.nnet.sigmoid(v)


def dropout(rng, x, p=0.5):
    seed = rng.randint(2 ** 30)
    srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
    mask = srng.binomial(n=1, p=1-p, size=x.shape)
    return x * T.cast(mask, theano.config.floatX)


class ReLU(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            #W_values *= 4  # TODO CHECK
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            #b_values = numpy.zeros((n_out,), dtype=theano.config.floatX) # TODO
            b_values = numpy.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        lin_output = T.dot(self.input, self.W) + self.b
        self.output = relu_f(lin_output)


class StackReLU(ReLU):
    def __init__(self, rng, input, in_stack, n_in, n_in_stack, n_out,
            W=None, Ws=None, b=None):
        self.input_stack = in_stack
        if Ws is None:
            Ws_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in_stack + n_out)),
                high=numpy.sqrt(6. / (n_in_stack + n_out)),
                size=(n_in_stack, n_out)), dtype=theano.config.floatX)
            #Ws_values *= 4  # TODO CHECK
            Ws = shared(value=Ws_values, name='Ws', borrow=True)
        self.Ws = Ws  # weights of the reccurrent connection
        super(StackReLU, self).__init__(rng, input, n_in, n_out)
        self.params = [self.W, self.b, self.Ws]  # order is important! W, b, Ws TODO that's because of adadelta not included here but in the nnet
        lin_output = (T.dot(self.input, self.W) 
                + T.dot(self.input_stack, self.Ws) + self.b)
        self.output = relu_f(lin_output)


class RecurrentReLU(StackReLU):
    def __init__(self, rng, input, in_stack, n_in, n_in_stack, n_out,
            W=None, Ws=None, b=None):
        super(RecurrentReLU, self).__init__(rng, input, n_in, n_out)
    # TODO (if needed)


class DropoutReLU(ReLU):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, dropout_rate=0.5):
        super(DropoutReLU, self).__init__(rng, input, n_in, n_out)
        self.dropout_rate = dropout_rate
        self.output = dropout(rng, self.output, self.dropout_rate)
