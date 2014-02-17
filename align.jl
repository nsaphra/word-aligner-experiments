module HMMWordAligners
import DataStructures.OrderedDict
import DataStructures.DefaultDict
import DataStructures.DefaultOrderedDict

abstract AbstractWordAligner
type HMMAligner{S<:String} <: AbstractWordAligner
    # e = target/emitter, f = source/emission
    f_vocab::Set{S}
    e_vocab::Set{S}

    # P(F|E)
    align_probs::Dict{S, Dict{S, Float64}}

    trans_probs::DefaultOrderedDict{Int, Float64, Float64}
    start_probs::Vector{Float64}

    function HMMAligner(f_vocab::Set{S}, e_vocab::Set{S};
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
        trans_probs = DefaultOrderedDict(0.0, [(j, trans_prob) for j=-jump_range:jump_range])
        trans_probs[1] = diag_prob

        start_prob = (1.0 - diag_prob) / (jump_range - 1)
        start_probs = [start_prob for j=1:jump_range]
        start_probs[1] = diag_prob

        new(f_vocab, e_vocab, align_probs, trans_probs, start_probs)
    end

    function HMMAligner(bitext::Vector{(Vector{S}, Vector{S})})
        f_vocab = Set{S}()
        e_vocab = Set{S}()
        for (f,e) = bitext
            union!(f_vocab, f)
            union!(e_vocab, e)
        end
        return HMMAligner{S}(f_vocab, e_vocab)
    end

    #TODO uncomment when foldl is in Base
    #HMMAligner(bitext::Vector{(Vector{S}, Vector{S})}) =
    #    HMMAligner(foldl((f_vocab, fe) -> union!(f_vocab, fe[1])::Set{S}, Set{S}(), bitext),
    #               foldl((e_vocab, fe) -> union!(e_vocab, fe[2])::Set{S}, Set{S}(), bitext))
end

# parameters to calculate expected counts:
# gamma[i,t] = P(F[t] = i| E, theta)
# digamma[i,j,t] = P(F[t] = i, F[t+1] = j| E, theta)
type HMMAlignerECounts{S}
    gamma_sum_no_last::Float64
    # sum_by_vocab is stored as e=>f=>prob to optimize for spatially localized hashing
    gamma_sum_by_vocab::Dict{S, Dict{S, Float64}}
    digamma_sum::Dict{Int, Float64}
    gammas_start::Vector{Float64}
end

macro bound_loop(ind, arr)
    quote
        if $ind < 1
            continue
         elseif $ind > length($arr)
            break
         end
    end
end

#TODO why won't julia let me specify a type for a?
function forward(a, f_sent, e_sent)
    fwd = zeros(Float64, length(f_sent), length(e_sent))

    for (f_ind, f) = enumerate(f_sent)
        # base case
        if f_ind == 1
            for (jmp, start_prob) = enumerate(a.start_probs)
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
            @bound_loop(e_ind, e_sent)

            fwd[f_ind, e_ind] += (a.align_probs[f][e_sent[e_ind]] * trans_prob * p)
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
            @bound_loop(e_ind, e_sent)

            bkw[f_ind, e_ind] += (a.align_probs[f_sent[f_ind+1]][e_sent[next_e_ind]]
                                  * trans_prob * p)
        end
    end
    return bkw
end

function expectation_step(c::HMMAlignerECounts, a, f_sent, e_sent)
    fwd = forward(a, f_sent, e_sent)
    bkw = backward(a, f_sent, e_sent)

    for (f_ind, f) = enumerate(f_sent)
        gamma_denom = 0.0
        digamma_denom = 0.0

        for (e_ind, e) = enumerate(e_sent)
            gamma_denom += fwd[f_ind, e_ind] * bkw[f_ind, e_ind]

            if (f_ind == length(f_sent))
                continue # last token doesn't give trans prob
            end
            for (jmp, trans_prob) = a.trans_probs
                e_ind2 = e_ind + jmp
                @bound_loop(e_ind2, e_sent)
                
                digamma_denom += fwd[f_ind, e_ind] * trans_prob * bkw[f_ind+1, e_ind2] *
                                 a.align_probs[f_sent[f_ind+1]][e_sent[e_ind2]]
            end
        end
        if digamma_denom == 0
            digamma_denom = 1e-9 # TODO logs
        end
        if gamma_denom == 0
            gamma_denom = 1e-9 # TODO logs
        end

        for (e_ind, e) = enumerate(e_sent)
            new_gamma = fwd[f_ind, e_ind] * bkw[f_ind, e_ind] / gamma_denom
            if (new_gamma == 0)
                continue
            end

            c.gamma_sum_by_vocab[e][f] += new_gamma
            if f_ind == 1
                c.gammas_start[e_ind] += new_gamma
            end

            if f_ind == length(f_sent)
                continue
            end
            # transition probs
            c.gamma_sum_no_last += new_gamma
            for (jmp, trans_prob) = a.trans_probs
                e_ind2 = e_ind + jmp
                @bound_loop(e_ind2, e_sent)

                c.digamma_sum[jmp] +=
                    fwd[f_ind, e_ind] * trans_prob *
                    bkw[f_ind+1, e_ind2] * a.align_probs[f_sent[f_ind+1]][e_sent[e_ind2]] /
                    digamma_denom
            end
        end
    end
end

function train{S}(a, bitext::Vector{(Vector{S}, Vector{S})}; numiter=100, epsilon=0.5)
    for iter = 1:numiter
        diff = 0.0
        if iter%10 == 0
            @printf(STDERR, "Training ... iter %i\n", iter)
        end

        counts = HMMAlignerECounts{S}(0.0,
                                      [e=>[f=>0.0::Float64 for f = a.f_vocab]::Dict{String, Float64} for e = a.e_vocab],
                                      [i=>0.0 for i = keys(a.trans_probs)],
                                      [0.0 for i = 1:length(a.start_probs)])
        for (n, (f_sent, e_sent)) = enumerate(bitext)
            expectation_step(counts, a, f_sent, e_sent)
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

        for jump = keys(a.trans_probs)
            old_p = a.trans_probs[jump]
            a.trans_probs[jump] = counts.digamma_sum[jump] / counts.gamma_sum_no_last
            diff += abs(a.trans_probs[jump] - old_p)
        end

        if iter%10 == 0
            @printf(STDERR, "diff %f", diff)
        end
        if diff < epsilon
            break
        end
    end
end

function decode(a, bitext)
    for (f_sent, e_sent) = bitext
        V = zeros(Float64, length(f_sent), length(e_sent))
        path = [Array(Int, 1) for e=e_sent]

        for (e_ind, start_prob) = enumerate(a.start_probs)
            if e_ind > length(e_sent)
                break
            end
            V[1, e_ind] = start_prob * a.align_probs[f_sent[1]][e_sent[e_ind]]
            path[e_ind] = [e_ind]
        end

        for (f_ind, f) = enumerate(f_sent)
            if f_ind == 1
                continue
            end

            new_path = [Array(Int, 1) for e=e_sent]
            for (e_ind, e) = enumerate(e_sent)
                (best_prob, best_prev_e) = findmax(
                    [V[f_ind-1, prev_e_ind] * a.trans_probs[e_ind - prev_e_ind] * a.align_probs[f][e]
                     for prev_e_ind = 1:length(V[f_ind-1, :])])
                V[f_ind, e_ind] = best_prob
                new_path[e_ind] = [path[best_prev_e], [e_ind]]
            end
            path = new_path
        end

        (p, end_e) = findmax([V[length(f_sent), e_ind] for e_ind=1:length(e_sent)])
        for (f_ind, e_ind) = enumerate(path[end_e])
             @printf "%i-%i " f_ind e_ind
        end
        println()
    end
end

function main(opts)
    f_data = @sprintf "%s.%s" opts["data"] opts["foreign"]
    e_data = @sprintf "%s.%s" opts["data"] opts["english"]

    bitext = [(split(strip(f)), split(strip(e)))
              for (f, e) = zip(open(readlines, f_data), open(readlines, e_data))]
    if length(bitext) > opts["num_sentences"]
        bitext = bitext[1:opts["num_sentences"]]
    end

    #TODO get rid of type param for constructor
    model = HMMAligner{String}(bitext)
    train(model, bitext, numiter=opts["num_iterations"])
    decode(model, bitext)
end

end # HMMWordAligners

using ArgParse
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--data", "-d"
            help = "Data filename prefix"
            default = "data/hansards"
        "--english", "-e"
            help = "Suffix of English filename"
            default = "e"
        "--foreign", "-f"
            help = "Suffix of Foreign filename"
            default = "f"
        "--num_sentences", "-n"
            help="Number of sentences (first in the file) to use for training and alignment"
            default = 1000
            arg_type = Int
        "--num_iterations", "-i"
            help = "Number of iterations to perform for EM"
            default = 100
            arg_type = Int
    end

    return parse_args(s)
end

opts = parse_commandline()
HMMWordAligners.main(opts)