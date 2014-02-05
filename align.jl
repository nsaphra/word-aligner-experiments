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

    function HMMAligner{S}(f_vocab::Vector{S}, e_vocab::Vector{S}; jump_range=10, diag_prob=0.5, init_dist="uniform")
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

    HMMAligner
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

end