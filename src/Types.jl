# Types.jl
"""
Lightweight types for the classical Hopfield practicum.

Defines:
- `MyClassicalHopfieldNetworkModel` - holds weight matrix `W`, bias `b`,
  energy dictionary for stored memories, and the original memories matrix.
"""

struct MyClassicalHopfieldNetworkModel
    W::Array{Float32,2}
    b::Array{Float32,1}
    energy::Dict{Int,Float32}
    memories::Union{Array{Int32,2}, Nothing}
end

"""Return a simple string summary for the model."""
Base.show(io::IO, m::MyClassicalHopfieldNetworkModel) =
    print(io, "MyClassicalHopfieldNetworkModel(N=", size(m.W,1), ", K=", (m.memories === nothing ? 0 : size(m.memories,2)), ")")
