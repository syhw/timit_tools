import sys, cPickle
from collections import defaultdict

unigrams = defaultdict(int)
bigrams = defaultdict(lambda: defaultdict(int))

def process(f):
    previous = None
    for line in f:
        if line[0].isdigit():
            current = line.rstrip('\n').split()[2]
            if previous != None:
                bigrams[previous][current] += 1
            unigrams[current] += 1
            previous = current
        else:
            previous = None
    s = sum(unigrams.itervalues())
    uni = dict(unigrams)
    bi = dict(bigrams)
    for phn in uni.iterkeys():
        uni[phn] *= 1.0 / s
    discounts = {}
    for phn, d in bi.iteritems():
        s = sum(d.itervalues())
        for phn2 in d.iterkeys():
            bi[phn][phn2] -= 0.5 # DISCOUNT
            bi[phn][phn2] *= 1.0 / s
        discounts[phn] = 1.0 - sum(bi[phn].itervalues())
    print uni
    print bi
    print discounts
    print sum(discounts.itervalues())

    with open('bigram.pickle', 'w') as of:
        cPickle.dump((uni, bi, discounts), of)
    print ">>> pickled bigram.pickle containing (unigrams, bigrams) dicts"

if len(sys.argv) < 2:
    print "python produce_LM.py train.mlf"

with open(sys.argv[1]) as f: 
    process(f)
