abstract type MyAbstractContextModel end
abstract type AbstractOnlineLearningModel end

"""
    mutable struct MyExperimentalDrugCocktailContext <: MyAbstractContextModel
A mutable type for the experimental context of the drug combination design problem.

### Fields
- `K::Int64`: number of drug types
- `m::Int64`: number of features per drug type
- `γ::Array{Float64,1}`: effectiveness parameters
- `μ::Array{Float64,1}`: mean parameters
- `B::Float64`: total budget in USD
- `cost::Dict{Int, Float64}`: maps drug type to cost per mg/kg
- `levels::Dict{Int, NamedTuple}`: maps drug level to drug concentration in mg/kg
"""
mutable struct MyExperimentalDrugCocktailContext <: MyAbstractContextModel
    
    # initialize -
    K::Int64               # number of drug types
    m::Int64               # number of features per drug type
    γ::Array{Float64,1}    # effectiveness parameters
    B::Float64             # total budget in USD
    cost::Dict{Int, Float64}      # maps drug type to cost per mg/kg
    levels::Dict{Int, NamedTuple} # maps drug level to drug concentration in mg/kg
    W::Float64             # weight of the patient in kg
    S::Float64             # safety constraint for maximum allowable dosage units: mg/kg-day
    bounds::Array{Float64,2}  # bounds for each drug type (L,U)

    # constructor -
    MyExperimentalDrugCocktailContext() = new(); # create new *empty* instance 
end

"""
    mutable struct MyQLearningAgentModel <: AbstractOnlineLearningModel

A mutable type for the Q-Learning Agent model.

### Fields
- `states::Array{Int,1}`: array of states
- `actions::Array{Int,1}`: array of actions
- `γ::Float64`: discount factor
- `α::Float64`: learning rate
- `Q::Array{Float64,2}`: Q-value table
"""
mutable struct MyQLearningAgentModel <: AbstractOnlineLearningModel

    # data -
    states::Array{Int,1}
    actions::Array{Int,1}
    γ::Float64
    α::Float64 
    Q::Array{Float64,2}

    # constructor
    MyQLearningAgentModel() = new();
end