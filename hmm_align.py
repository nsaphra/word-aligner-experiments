import sys
import optparse
from collections import defaultdict
from math import log
import numpy as np
import re

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
            total_cnts = defaultdict(float)
            self.align_probs = {f:defaultdict(float) for f in self.f_vocab}
            for (f_sent, e_sent) in bitext:
                for f in f_sent:
                    for e in e_sent:
                        total_cnts[f] += 1
                        self.align_probs[f][e] += 1
            for (f, e_probs) in self.align_probs.items():
                for e in e_probs.keys():
                    e_probs[e] /= total_cnts[f]

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

    def forward(self, e_sent, f_sent, alignments={}):
        fwd = [0] * len(f_sent)
        for (f_ind, f) in enumerate(f_sent): # target / input
            fwd[f_ind] = defaultdict(float)
            if f_ind in alignments:
                for e_ind in alignments[f_ind]:
                    fwd[f_ind][e_ind] = 1.0/len(alignments[f_ind])
                continue

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

    def backward(self, e_sent, f_sent, alignments={}):
        bkw = [0] * len(f_sent)
        for (rev_f_ind, f) in enumerate(reversed(f_sent)):
            f_ind = len(f_sent) - rev_f_ind - 1
            bkw[f_ind] = defaultdict(float)
            if f_ind in alignments:
                for e_ind in alignments[f_ind]:
                    bkw[f_ind][e_ind] = 1.0/len(alignments[f_ind])
                continue

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

    def expectation_step(self, e_sent, f_sent, gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum, alignments=None):
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
    def train(self, bitext, numiter=100, epsilon=5, likely_align_iter=10, fixed_a=None, all_a=None):
        print >> sys.stderr, "Training ..."
        for i in range(numiter):
            diff = 0.0
            if i%5 == 0:
                print >> sys.stderr, "Iteration %i"%i
            gamma_sum_no_last = 0.0
            gamma_sum_by_vocab = dict((e, defaultdict(float)) for e in self.e_vocab)
            digamma_sum = defaultdict(float)
            gammas_0s = defaultdict(float)

            for (n, (f_sent, e_sent)) in enumerate(bitext):
                alignments = None
                if n < len(all_a):
                    if i < likely_align_iter:
                        alignments = all_a
                    else:
                        alignments = fixed_a
                    
                (gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum) = \
                    self.expectation_step(e_sent, f_sent,\
                                              gammas_0s, gamma_sum_no_last, gamma_sum_by_vocab, digamma_sum,\
                                              alignments=alignments)

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
            V = [[0.0 for e in e_sent] for f in f_sent]
            path = {}

            for (e_ind,e) in enumerate(e_sent):
                if e_ind in self.start_probs:
                    V[0][e_ind] = self.start_probs[e_ind] * self.align_probs[f_sent[0]][e_sent[e_ind]]
                    path[e_ind] = [e_ind]

            for (f_ind, f) in enumerate(f_sent):
                if f_ind == 0:
                    continue
                new_path = {}

                for (e_ind, e) in enumerate(e_sent):
                    best_prob = 0.0
                    best_prev_e = 0

                    for prev_e_ind in range(len(V[f_ind-1])):
                        jump = e_ind - prev_e_ind
                        if jump in self.trans_probs and V[f_ind-1][prev_e_ind]:
                            trans_prob = self.trans_probs[jump]
                            align_prob = self.align_probs[f][e]
                            prob_c = V[f_ind-1][prev_e_ind] * \
                                trans_prob * \
                                align_prob
                            if prob_c > best_prob:
                                best_prob = prob_c
                                best_prev_e = prev_e_ind
                    if best_prob:
                        V[f_ind][e_ind] = best_prob
                        new_path[e_ind] = path[best_prev_e] + [e_ind]
                path = new_path

            (p, end_e) = max([(V[len(f_sent) - 1][e_ind], e_ind) for e_ind in range(len(e_sent))], key=operator.itemgetter(0))
            alignments = ""
            for (f_ind, e_ind) in enumerate(path[end_e]):
                alignments += "%i-%i " % (f_ind, e_ind)
            print(alignments)

if __name__ == "__main__":
    # args ripped from dreamt aligner
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-i", "--num_iterations", dest="numiter", default=100, type="int", help="Number of iterations to perform for EM")
    optparser.add_option("-p", "--model_pickle", dest="pickle_file", default="", help="File to store pickle of model if desired")
    optparser.add_option("-a", "--alignments", dest="aligned", default="a", help="Suffix of alignments file (default=a)")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)
    a_data = "%s.%s" % (opts.train, opts.aligned)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    model = HMMAligner(bitext)

    fixed_alignments=[sentence.strip().split() for sentence in open(a_data)]
    fixed_a = [{} for s in fixed_alignments]
    all_a = [{} for s in fixed_alignments]
    if fixed_alignments:
        for (i, sentence) in enumerate(fixed_alignments):
            for a in sentence:
                m = re.search(r"(\d+)(\?|-)(\d+)", a)
                f_tok = int(m.group(1))
                e_tok = int(m.group(3))
                known = (m.group(2) == "-")
                if known:
                    if f_tok not in fixed_a:
                        fixed_a[i][f_tok] = {e_tok}
                    else:
                        fixed_a[i][f_tok].add(e_tok)
                if f_tok not in all_a:
                    all_a[i][f_tok] = {e_tok}
                else:
                    all_a[i][f_tok].add(e_tok)

    model.train(bitext, opts.numiter, fixed_a=fixed_a, all_a=all_a)
    if opts.pickle_file:
        pickle.dump(model, open(opts.pickle_file,'wb'))
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
