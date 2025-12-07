# Types are loaded into Main by Include.jl via include("src/Types.jl");
# refer to `MyClassicalHopfieldNetworkModel` directly

"""Build a `MyClassicalHopfieldNetworkModel` from keyword args.

Accepted keyword arguments:
- `memories` :: Array{Int32,2} (N x K) where each column is a pattern in {-1,1}

The weights are computed using the (unnormalized) Hebbian rule and biases set to zero.
"""
function build(::Type{MyClassicalHopfieldNetworkModel}; memories)
    N, K = size(memories)
    W = zeros(Float32, N, N)
    # Hebbian outer product sum (average over K)
    for k in 1:K
        s = Float32.(memories[:,k])
        W .+= (s * s')
    end
    W ./= Float32(K)
    # zero out diagonal (no self-connections)
    for i in 1:N
        W[i,i] = 0.0f0
    end
    b = zeros(Float32, N)

    # compute energy of each stored memory for reference
    energy = Dict{Int,Float32}()
    for k in 1:K
        s = Float32.(memories[:,k])
        E = -0.5f0 * dot(s, W * s) - dot(b, s)
        energy[k] = E
    end

    return MyClassicalHopfieldNetworkModel(W, b, energy, memories)
end

# compatibility: accept a NamedTuple like (memories = mat,) used in the notebook
function build(::Type{MyClassicalHopfieldNetworkModel}, args::NamedTuple)
    if haskey(args, :memories)
        return build(MyClassicalHopfieldNetworkModel; memories = args[:memories])
    else
        throw(ArgumentError("build(MyClassicalHopfieldNetworkModel, args) expects a NamedTuple with key :memories"))
    end
end
