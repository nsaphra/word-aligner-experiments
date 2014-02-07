module HMMWordAligners
import DataStructures.OrderedDict
import DataStructures.DefaultDict

abstract AbstractWordAligner
type HMMAligner{S<:String} <: AbstractWordAligner
    # e = target/emitter, f = source/emission
    f_vocab::Vector{S}
    e_vocab::Vector{S}

    # P(F|E)
    align_probs::Dict{S, Dict{S, Float64}}

    trans_probs::OrderedDict{Int, Float64}
    start_probs::OrderedDict{Int, Float64}

    function HMMAligner{S}(f_vocab::Vector{S}, e_vocab::Vector{S};
                           jump_range=10, diag_prob=0.5, init_dist="uniform")
        align_probs
        if init_dist == "uniform"
            align_prob = (1.0 / length(f_vocab))::Float64
            align_probs = [f=>[e=>align_prob for e=e_vocab]::Dict{S, Float64} for f=f_vocab]
        else
            error("Non-uniform initialize not implemented.")
        end

        trans_prob = ((1.0 - diag_prob) / (jump_range * 2))::Float64
#        trans_probs = OrderedDict{Int, Float64}()
        trans_probs = OrderedDict([(j, trans_prob) for j=-jump_range:jump_range])::OrderedDict{Int, Float64}
        trans_probs[1] = diag_prob

        start_prob = (1.0 - diag_prob) / (jump_range - 1)
        start_probs = OrderedDict([(j, start_prob) for j=1:jump_range])
        start_probs[1] = diag_prob

        new(f_vocab, e_vocab, align_probs, trans_probs, start_probs)
    end

    HMMAligner(f_vocab, e_vocab;
               jump_range=10, diag_prob=0.5, init_dist="uniform") =
            HMMAligner{eltype(f_vocab)}(f_vocab, e_vocab, jump_range, diag_prob, init_dist)

    HMMAligner(bitext) = HMMAligner(reduce((fe, f_vocab) -> union(set(fe[1]), f_vocab), bitext, set()),
                                    reduce((fe, e_vocab) -> union(set(fe[2]), e_vocab), bitext, set()))
end

# parameters to calculate expected counts:
# gamma[i,t] = P(F[t] = i| E, theta)
# digamma[i,j,t] = P(F[t] = i, F[t+1] = j| E, theta)
type HMMAlignerECounts{S}
    gamma_sum_no_last::Float64
    # sum_by_vocab is stored as e=>f=>prob to optimize for spatially localized hashing
    gamma_sum_by_vocab::Dict{S, Dict{S, Float64}}
    digamma_sum::Vector{Float64}
    gammas_start::Vector{Float64}
end


#TODO why won't julia let me specify a type for a?
function forward(a, f_sent, e_sent)
    fwd = zeros(Float64, length(f_sent), length(e_sent))

    for (f_ind, f) = enumerate(f_sent)
        # base case
        if f_ind == 1
            for (jmp, start_prob) = a.start_probs
                if jmp > length(e_sent)
                    break
                end
                fwd[f_ind, jmp] = (a.start_probs[jmp] * a.align_probs[f][e_sent[jmp]])
            end
            continue
        end

        for (jmp, trans_prob) = a.trans_probs,
            (prev_e_ind, p) = enumerate(fwd[f_ind - 1,:])
            e_ind = prev_e_ind + jmp
            if e_ind > length(e_sent)
                break
            end
            if e_ind >= 1 # in sentence bounds
                fwd[f_ind, e_ind] += (a.align_probs[f][e_sent[e_ind]] * trans_prob * p)
            end
        end
    end
    return fwd
end

function backward(a, f_sent, e_sent)
    bkw = zeros(Float64, length(f_sent), length(e_sent))

    for (rev_f_ind, f) = enumerate(reverse(f_sent))
        f_ind = length(f_sent) - rev_f_ind + 1

        # base case
        if f_ind == length(f_sent)
            #TODO add EOS probs
            for e_ind = 1:length(e_sent)
                bkw[f_ind, e_ind] = 1.0
            end
            continue
        end

        for (jmp, trans_prob) = a.trans_probs,
            (next_e_ind, p) = enumerate(bkw[f_ind + 1,:])
            e_ind = next_e_ind - jmp
            if e_ind > length(e_sent)
                break
            end
            if e_ind >= 1
                bkw[f_ind, e_ind] += (a.align_probs[f_sent[f_ind+1]][e_sent[next_e_ind]]
                                      * trans_prob * p)
            end
        end
    end
    return bkw
end

function expectation_step(c::HMMAlignerECounts, a, f_sent, e_sent)
    fwd = forward(a, f_sent, e_sent)
    bkw = backward(a, f_sent, e_sent)

    for (f_ind, f) = enumerate(f_sent)
        gamma_denom = 0.0
        for (e_ind, e) = enumerate(e_sent)
            gamma_denom += fwd[f_ind][e_ind] * bkw[f_ind][e_ind]

            if (f_ind >= length(f_sent)-1)
                continue
            end
            for (e_ind2, e2) = enumerate(e_sent)
                digamma_denom += fwd[f_ind][e_ind] * a.trans_probs[e_ind2 - e_ind] *
                                 bkw[f_ind+1][e_ind2] * a.align_probs(f_sent[f_ind+1], e2)
            end
            if digamma_denom == 0
                digamma_denom = 1e-9
            end
        end

        for (e_ind, e) = enumerate(e_sent)
            if gamma_denom == 0
                gamma_denom = 1e-9
            end

            new_gamma = fwd[f_ind, e_ind] * bkw[f_ind, e_ind] / gamma_denom
            c.gamma_sum_by_vocab[e][f] += new_gamma

            if f_ind == 1
                c.gammas_start[e_ind] += new_gamma
            end
            if f_ind >= length(f_sent)
                continue
            end

            # transition probs
            c.gamma_sum_no_last += new_gamma
            for (e_ind2, e2) = enumerate(e_sent)
                c.digamma_sum[e_ind2 - e_ind] +=
                    fwd[f_ind, e_ind] * self.trans_probs[e_ind2 - e_ind] *
                    bkw[f_ind+1, e_ind2] * self.smoothed_align_probs(f_sent[f_ind+1], e2) /
                    digamma_denom
            end
        end
    end
end

function train(a, bitext; numiter=100, epsilon=0.5)
    for iter = 1:numiter
        diff = 0.0
        if i%10 == 0
            write(STDERR, "Training")
        end

        counts = HMMAlignerECounts(0.0,
                                   [e=>[f=>0.0 for f = a.f_vocab] for e = a.e_vocab],
                                   [i=>0.0 for i = keys(a.trans_probs)],
                                   [0.0 for i = 1:length(a.start_probs)])
        for (n, (f_sent, e_sent)) = enumerate(bitext)
            self.expectations_step(counts, a, f_sent, e_sent)
        end

        a.start_probs = counts.gammas_start

        # compute MLE params to model
        for (e, fs) = counts.gamma_sum_by_vocab
            prob_e = sum(values(fs))
            for (f, prob_ef) = fs
                old_p = a.align_probs[f][e]
                a.align_probs[f][e]= 0
                if prob_ef != 0
                    a.align_probs[f][e] = prob_ef / prob_e
                end
                diff += abs(a.align_probs[f][e] - old_p)
            end
        end

        for jump in keys(a.trans_probs)
            old_p = a.trans_probs[jump]
            a.trans_probs[jump] = counts.digamma_sum[jump] / counts.gamma_sum_no_last
            diff += abs(a.trans_probs[jump] - old_p)
        end

        if diff < epsilon
            break
        end
    end
end

function decode(a, bitext)
    error("NOT IMPLEMENTED")
end

end