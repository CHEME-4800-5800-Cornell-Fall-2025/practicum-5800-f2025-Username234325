abstract type MyAbstractContextModel end
abstract type AbstractOnlineLearningModel end

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