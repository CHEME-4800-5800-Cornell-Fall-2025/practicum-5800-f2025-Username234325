# Types are loaded into Main by Include.jl via include("src/Types.jl");
using Random

"""Decode a state vector s (elements ±1) into a matrix with `nrows` x `ncols`.

If `nrows`/`ncols` are not provided the function will attempt to make a square
matrix when possible.
"""
function decode(s::AbstractVector{<:Integer}; nrows::Union{Int,Nothing}=nothing, ncols::Union{Int,Nothing}=nothing)
    N = length(s)
    if nrows === nothing && ncols === nothing
        r = Int(round(sqrt(N)))
        c = div(N, r)
    elseif nrows === nothing
        c = ncols
        r = div(N, c)
    elseif ncols === nothing
        r = nrows
        c = div(N, r)
    else
        r = nrows; c = ncols
    end
    # Return Float32 values in [0,1] so they convert cleanly to Gray types.
    M = Array{Float32,2}(undef, r, c)
    idx = 1
    for i in 1:r, j in 1:c
        v = s[idx]
        # map {-1, 1} -> {0.0, 1.0}
        M[i,j] = v == Int32(-1) ? 0.0f0 : 1.0f0
        idx += 1
    end
    return M
end

"""Compute Hamming distance between two integer vectors (counts differing entries)."""
function hamming(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    @assert length(a) == length(b)
    cnt = 0
    for i in 1:length(a)
        if a[i] != b[i]
            cnt += 1
        end
    end
    return cnt
end

"""Sign function that returns ±1. On zero returns +1."""
sign1(x::Real) = x >= 0 ? Int32(1) : Int32(-1)

"""Compute energy of a state `s` for a given model."""
function energy(model::MyClassicalHopfieldNetworkModel, s::AbstractVector{<:Integer})
    s_f = Float32.(s)
    return -0.5f0 * dot(s_f, model.W * s_f) - dot(model.b, s_f)
end

"""Run asynchronous recover algorithm on `model` starting from s0.

Returns (frames, energydictionary) where frames maps step index to state vector
and energydictionary maps step index to energy. The algorithm uses random
single-site updates and a patience criterion for convergence.
"""
function recover(model::MyClassicalHopfieldNetworkModel, s0::Array{Int32,1}, true_image_energy::Float32; maxiterations::Int=1000, patience::Union{Int,Nothing}=5, miniterations_before_convergence::Union{Int,Nothing}=nothing)
    N = length(s0)
    # defaults
    if patience === nothing
        patience = max(5, N) # fallback
    end
    if miniterations_before_convergence === nothing
        miniterations_before_convergence = patience
    end

    frames = Dict{Int, Array{Int32,1}}()
    energydictionary = Dict{Int, Float32}()

    # current state
    s = copy(s0)
    frames[1] = copy(s)
    energydictionary[1] = energy(model, s)

    # queue of past states for patience check (lightweight vector, avoids external deps)
    S = Vector{Array{Int32,1}}()
    push!(S, copy(s))

    converged = false
    t = 1
    rng = Random.GLOBAL_RNG

    while (!converged) && (t < maxiterations)
        t += 1
        # pick a random neuron (index)
        i = rand(rng, 1:N)
        # local field
        h = sum(Float32.(model.W[i, :]) .* Float32.(s)) - model.b[i]
        s[i] = sign1(h)

        frames[t] = copy(s)
        energydictionary[t] = energy(model, s)

        push!(S, copy(s))
        if length(S) > patience
            popfirst!(S)
        end

        # check patience if we've run enough iterations
        if t >= miniterations_before_convergence
            all_same = true
            if length(S) == patience
                first = S[1]
                for x in S
                    if !all(x .== first)
                        all_same = false; break
                    end
                end
            else
                all_same = false
            end

            if all_same
                converged = true
            end
        end

        # check if we've hit the true image energy (or lower)
        if energydictionary[t] <= true_image_energy
            converged = true
        end
    end

    return frames, energydictionary
end
