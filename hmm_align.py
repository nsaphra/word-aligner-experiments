import sys
import optparse
from collections import defaultdict
from math import log
import numpy as np

class HMMAligner:
    """An HMM word alignment model"""
    def __init__(self, bitext, align_probs=None, trans_probs=None, start_probs=None):
        # e = target/input, f = source/emission
        self.f_vocab = reduce(lambda x,y: x.union(set(y[0])), bitext, set())
        self.e_vocab = reduce(lambda x,y: x.union(set(y[1])), bitext, set())

        if align_probs:
            self.align_probs = align_probs
            # TODO make this part work if necessary
        else:
            dim = len(self.e_vocab) * len(self.f_vocab)
            self.align_probs = {}
            for f in self.f_vocab:
                self.align_probs[f] = {}
                for e in self.e_vocab:
                    # p(f|e)
                    self.align_probs[f][e] = 1.0 / dim

        # frumious hack, TODO better initialization
        # Initialize with higher diagonal probabilities.
        if trans_probs and start_probs:
            self.trans_probs = trans_probs
        else:
            self.trans_probs = defaultdict(float)
            for i in range(-10, 10):
                self.trans_probs[i] = 0.5 / 19
            self.trans_probs[1] = 0.5
        if start_probs:
            self.start_probs = start_probs
        else:
            self. start_probs = defaultdict(float)
            for i in range(10):
                self.start_probs[i] = 0.5 / 9
            self.start_probs[0] = 0.5

    def forward(self, e_sent, f_sent):
        fwd = [0] * len(f_sent)
        for (f_ind, f) in enumerate(f_sent): # target / input
            fwd[f_ind] = defaultdict(float)
            # base case
            if f_ind == 0:
                for jump in self.start_probs.keys():
                    if jump >= len(e_sent):
                        continue
                    fwd[f_ind][jump] = self.start_probs[jump] * self.align_probs[f][e_sent[jump]]
                continue

            for jump in self.trans_probs.keys(): # alignment state
                for (prev_e_ind, p) in fwd[f_ind - 1].items():
                    e_ind = prev_e_ind + jump
                    if e_ind >= len(e_sent) or e_ind < 0:
                        continue
                    fwd[f_ind][e_ind] += self.align_probs[f][e_sent[e_ind]] * p * self.trans_probs[jump]
        return fwd

    def backward(self, e_sent, f_sent):
        bkw = [0] * len(f_sent)
        for (rev_f_ind, f) in enumerate(reversed(f_sent)):
            f_ind = len(f_sent) - rev_f_ind - 1
            bkw[f_ind] = defaultdict(float)

            # base case
            if rev_f_ind == 0:
                for (e_ind, e) in enumerate(e_sent):
                    bkw[f_ind][e_ind] = 1
                continue

            for jump in self.trans_probs.keys():
                for (next_e_ind, p) in bkw[f_ind + 1].items():
                    e_ind = next_e_ind - jump
                    if e_ind >= len(e_sent) or e_ind < 0:
                        continue
                    bkw[f_ind][e_ind] += self.align_probs[f_sent[f_ind+1]][e_sent[next_e_ind]] \
                        * self.trans_probs[jump] \
                        * p
        return bkw

    def expectation_step(self, e_sent, f_sent, gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum):
        fwd = self.forward(e_sent, f_sent)
        bkw = self.backward(e_sent, f_sent)

        for (f_ind, f) in enumerate(f_sent):
            gamma_denom = 0.0
            digamma_denom = 0.0

            # compute expected probs (gamma, digamma)
            for (e_ind, e) in enumerate(e_sent):
                gamma_denom += fwd[f_ind][e_ind] * bkw[f_ind][e_ind]

                if (f_ind >= len(f_sent)-1): # last token
                    continue
                # else < f - 1, so digamma needed for trans probs
                for (e_ind2, e2) in enumerate(e_sent):
                    digamma_denom += fwd[f_ind][e_ind] * self.trans_probs[e_ind2 - e_ind] \
                        * bkw[f_ind+1][e_ind2] * self.align_probs[f_sent[f_ind+1]][e2]

#                digamma_denom += \
#                    sum(fwd[f_ind][e_ind] * self.trans_probs[e_ind2 - e_ind] \
#                            * bkw[f_ind+1][e_ind2] * self.align_probs[f_sent[f_ind+1]][e2] \
#                            for (e_ind2, e2) in enumerate(e_sent))

            for (e_ind, e) in enumerate(e_sent):
                # temporary for testing. will switch to log probs. TODO
                if gamma_denom == 0:
                    print >> sys.stderr, "FIX ME I HAVE A 0"
                    return (gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum)
                # TODO bug? shouldn't be 0 ever
                new_gamma = fwd[f_ind][e_ind] * bkw[f_ind][e_ind] / gamma_denom
                gamma_sum_by_vocab[e][f] += new_gamma

                if (f_ind == 0):
                    gammas_0s[e_ind] += new_gamma

                if (f_ind >= len(f_sent)-1):
                    continue
                # else < f - 1, so do transition probs
                gamma_sum_no_last += new_gamma
                for (e_ind2, e2) in enumerate(e_sent):
                    digamma_sum[e_ind2 - e_ind] += \
                        fwd[f_ind][e_ind] * self.trans_probs[e_ind2 - e_ind] \
                        * bkw[f_ind+1][e_ind2] * self.align_probs[f_sent[f_ind+1]][e2] \
                        / digamma_denom

        return (gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum)

    # bitext should be [target, source]
    def train(self, bitext, numiter=100, epsilon=0.5):
        print >> sys.stderr, "Training ..."
        for i in range(numiter):
            diff = 0.0
            if i%5 == 0:
                print >> sys.stderr, "Iteration %i"%i
            # gamma_i(t) = P(X_t = i | Y, theta)
            gamma_sum_no_last = 0.0
            gamma_sum_by_vocab = dict((e, defaultdict(float)) for e in self.e_vocab)
            # digamma_{i,j}(t) = P(X_t = i, X_{t+1} = j | Y, theta)
            digamma_sum = defaultdict(float)
            gammas_0s = defaultdict(float)

            for (n, (f_sent, e_sent)) in enumerate(bitext):
                (gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum) = \
                    self.expectation_step(e_sent, f_sent,\
                                              gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum)

            # compute maximum-likelihood params
            # TODO add start probs to the difference count
            self.start_probs = gammas_0s

            for (e, fs) in gamma_sum_by_vocab.items():
                prob_e = sum(fs.values())
                for (f, prob_ef) in fs.items():
                    old_p = self.align_probs[f][e]
                    self.align_probs[f][e] = prob_ef / prob_e
                    diff += abs(self.align_probs[f][e] - old_p)

            for jump in self.trans_probs.keys():
                old_p = self.trans_probs[jump]
                self.trans_probs[jump] = digamma_sum[jump] / gamma_sum_no_last
                diff += abs(self.trans_probs[jump] - old_p)

            if i%5 == 0:
                print >> sys.stderr, diff
            if diff < epsilon:
                break

    def decode(self, bitext):
        for (f_sent, e_sent) in bitext:
            alignments = ""
            V = [defaultdict(float) for f in f_sent]
            path = {}

            for e_ind in self.start_probs.keys():
                if e_ind >= len(e_sent):
                    break
                V[0][e_ind] = self.start_probs[e_ind] * self.align_probs[f_sent[0]][e_sent[e_ind]]
                path[e_ind] = [e_ind]

            for (f_ind, f) in enumerate(f_sent):
                if (f_ind == 0):
                    continue # TODO seriously get rid of all these nops on 0 / end inds
                new_path = {}

                for (e_ind, e) in enumerate(e_sent):
                    (best_prob, best_prev_e) = max(\
                        (V[f_ind-1][prev_e_ind] * self.trans_probs[e_ind - prev_e_ind] * self.align_probs[f][e], \
                             prev_e_ind) \
                            for prev_e_ind in V[f_ind-1].keys())
                    V[f_ind][e_ind] = best_prob
                    new_path[e_ind] = path[best_prev_e] + [e_ind]
                path = new_path

            (p, end_e) = max((V[len(f_sent) - 1][e_ind], e_ind) for e_ind in range(len(e_sent)))
            for (f_ind, e_ind) in enumerate(path[end_e]):
                alignments += "%i-%i " % (f_ind, e_ind)
            print(alignments)

if __name__ == "__main__":
    # args ripped from dreamt aligner
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-i", "--num_iterations", dest="numiter", default=100, type="int", help="Number of iterations to perform for EM")
    optparser.add_option("-p", "--model_pickle", dest="pickle_file", default="", help="File to store pickle of model if desired")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    model = HMMAligner(bitext)

    model.train(bitext, opts.numiter)
    if pickle_file:
        pickle.dump(model, open(pickle_file,'wb'))
    model.decode(bitext)

def test_fwd_bkw():
    # fwd
    # a: [1:0.375, 2:0.125, 3:0 ]
    # b: [1:0.0279, 2:0.065175, 3:0.01875]
    # c: [1:0.0020925, 2:0.009073125, 3:0.022365]
    # bkw
    # c: [1:1, 2:1, 3:1]
    # b: [1:0.225, 2:0.375, 3:0.15]
    # a: [1:0.073, 2:0.052, 3:0.011]
    e_sent = ["1", "2", "3"]
    f_sent = ["a", "b", "c"]
    align_probs = defaultdict(float, {"a":{"1":0.5, "2":0.5, "3":0.25},\
                                          "b":{"1":0.25, "2":0.25, "3":0.25},\
                                          "c":{"1":0.25, "2":0.25, "3":0.5}})
    trans_probs = defaultdict(float, {0:0.3, 1:0.6})
    start_probs = defaultdict(float, {0:0.75, 1:0.25})
    m = HMMAligner((e_sent, f_sent), align_probs, trans_probs, start_probs)
    return (m.forward(e_sent, f_sent), m.backward(e_sent, f_sent))
