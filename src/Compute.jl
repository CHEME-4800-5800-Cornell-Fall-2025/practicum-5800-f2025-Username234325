function solve(agent::MyQLearningAgentModel, environment::MyExperimentalDrugCocktailContext; 
    maxsteps::Int = 100, δ::Float64 = 0.02, worldmodel::Function = _world)::MyQLearningAgentModel

    # initialize -
    actions = agent.actions;
    K = length(actions); # number of actions
    states = agent.states;
    Q₁ = agent.Q;
    γ = agent.γ;

    # simulation loop -
    for s ∈ states

        # initialize t -
        t = 1; # trial counter
        has_converged = false;
        αₜ = copy(agent.α);   
        while (has_converged == false)
            
            # compute the ϵ -
            ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3); # compute the epsilon value -
            p = rand();

            aₜ = nothing;
            if p ≤ ϵₜ
                aₜ = rand(1:K); # generate a random action
            elseif p > ϵₜ
                aₜ = argmax(Q₁[s,:]); # select the greedy action, given state s
            end

            # compute new state and reward -
            s′, r = worldmodel(environment, s, aₜ);
            
            # use the update rule to update Q -
            Q₂ = copy(Q₁); # this seems really inefficient, but it is what it is ...
            Q₁[s,aₜ] += αₜ*(r+γ*maximum(Q₁[s′,:]) - Q₁[s,aₜ])

            # update stuff
            s = s′; # state update
            t += 1; # trial update
            αₜ = 0.99*αₜ; # update the learning rate

            # check if we have converged -
            if ((t ≥ maxsteps) || norm(Q₂ - Q₁) ≤ δ)
                has_converged = true;

                # warn if we hit maxsteps -
                if (t ≥ maxsteps) && (norm(Q₂ - Q₁) > δ)
                    @warn "Maximum number of steps reached before convergence."
                end
            end
        end
    end

    agent.Q = Q₁; # update the model

    # return -
    return agent
end

"""
    mypolicy(Q_array::Array{Float64,2}) -> Array{Int,1}

This function computes the policy from the Q-value function.

### Arguments
- `Q_array::Array{Float64,2}`: the Q-value function

### Returns
- `Array{Int,1}`: the policy which maps states to actions
"""
function mypolicy(Q_array::Array{Float64,2})::Array{Int64,1}

    # get the dimension -
    (NR, _) = size(Q_array);

    # initialize some storage -
    π_array = Array{Int64,1}(undef, NR)
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end

    # return -
    return π_array;
end