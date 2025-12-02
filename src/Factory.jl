function build(modeltype::Type{MyExperimentalDrugCocktailContext}, data::NamedTuple)::MyExperimentalDrugCocktailContext
    
    # create new context instance -
    context = modeltype() # model is empty instance

    # populate context fields from data -
    context.K = data.K
    context.m = data.m
    context.γ = data.γ
    context.B = data.B
    context.cost = data.cost
    context.levels = data.levels
    context.W = data.W
    context.S = data.S
    context.bounds = data.bounds

    return context
end

function build(modeltype::Type{MyQLearningAgentModel}, data::NamedTuple)::MyQLearningAgentModel
    
    # create new model instance -
    model = modeltype()

    # populate model fields from data -
    model.states = data.states
    model.actions = data.actions
    model.γ = data.γ
    model.α = data.α
    model.Q = data.Q

    return model
end