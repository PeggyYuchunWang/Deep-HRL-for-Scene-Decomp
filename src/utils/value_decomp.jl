include("../AutomotiveHRLSceneDecomp.jl")

struct ComposedPolicy <: AbstractNNPolicy
    lane_change_policy::NNPolicy
    intersect_policy::NNPolicy
end

function action_values(p::ComposedPolicy, s::Scene)
    return 0
end

function action_values(p::ComposedPolicy, v::V) where V<:AbstractArray
    s_lc = v[1]
    s_in = v[2]
    return action_values(p.lane_change_policy, s_lc) + action_values(p.intersect_policy, s_in)
end
