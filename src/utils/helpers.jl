include("../AutomotiveHRLSceneDecomp.jl")

function collision_helper(s::Scene, mdp::DrivingMDP)
    ego = s[findfirst(mdp.ego_id, s)]
    for veh in s
        if veh.id != mdp.ego_id
            if collision_checker(ego, veh)
                return true
            end
        end
    end
    return false
end

function off_road(s::Scene, mdp::DrivingMDP)
    ego = s[findfirst(mdp.ego_id, s)]
    if abs(ego.state.posF.t) >= 1.5
        return true
    end
    return false
end

function distance(s::Scene, mdp::DrivingMDP)
    ego = s[findfirst(mdp.ego_id, s)]
    goal = get_posG(mdp.goal_pos, mdp.roadway)
    d = norm(VecE2(goal - ego.state.posG))
    return d
end

function reachgoal(s::Scene, mdp::DrivingMDP)
    ego = s[findfirst(mdp.ego_id, s)]
    if mdp.goal_pos.roadind.tag == ego.state.posF.roadind.tag && ego.state.posF.s >= mdp.road_length && abs(mdp.goal_pos.t-ego.state.posF.t) <= 0.5
        return true
    end
    return false
end

function safe_actions(mdp::DrivingMDP, s::Scene)
    safe_acts = []
    for a in actions(mdp)
        sp = deepcopy(s)
        action_list = LatLonAccel[LatLonAccel(0.,0.0) for veh in s if veh.id != mdp.ego_id]
        tick!(sp, mdp.roadway, [a, action_list...], mdp.delta_t)
        ego = sp[findfirst(mdp.ego_id, sp)]
        if !off_road(sp, mdp) && !collision_helper(sp, mdp) && ego.state.v < mdp.speed_limit
            push!(safe_acts, a)
        end
    end
    return safe_acts
end

function safe_actions(mdp::DrivingMDP, o::AbstractArray)
    s = POMDPs.convert_s(Scene, o, mdp)
    return safe_actions(mdp, s)
end

function best_action(acts::Vector{A}, val::AbstractArray{T}, problem::M) where {A, T <: Real, M <: Union{POMDP, MDP}}
    best_a = acts[1]
    best_ai = actionindex(problem, best_a)
    best_val = val[best_ai]
    for a in acts
        ai = actionindex(problem, a)
        if val[ai] > best_val
            best_val = val[ai]
            best_ai = ai
            best_a = a
        end
    end
    return best_a::A
end

function masked_linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float64, eps_end::Float64)
    # define function that will be called to select an action in DQN
    # only supports MDP environments
    function action_masked_epsilon_greedy(policy::AbstractNNPolicy, env::MDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
        eps = DeepQLearning.update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        acts = safe_actions(mdp, obs) #XXX using pomdp global variable replace by safe_actions(mask, obs)
        val = actionvalues(policy, obs) #change this
        if rand(rng) < eps
            return (rand(rng, acts), eps)
        else
            return (best_action(acts, val, env.problem), eps)
        end
    end
    return action_masked_epsilon_greedy
end

function action_masked_epsilon_greedy(policy::AbstractNNPolicy, env::MDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
    eps = DeepQLearning.update_epsilon(global_step, eps_fraction, eps_end, max_steps)
    acts = safe_actions(mdp, obs) #XXX using pomdp global variable replace by safe_actions(mask, obs)
    val = actionvalues(policy, obs)
    if rand(rng) < eps
        return (rand(rng, acts), eps)
    else
        return (best_action(acts, val, env.problem), eps)
    end
end
