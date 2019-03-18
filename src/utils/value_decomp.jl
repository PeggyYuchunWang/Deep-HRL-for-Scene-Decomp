include("../AutomotiveHRLSceneDecomp.jl")
include("../mdps/composition_intersection.jl")
include("../mdps/simple_intersection.jl")
include("../mdps/simple_two_lane.jl")
include("helpers.jl")

struct ComposedPolicy <: AbstractNNPolicy
    lane_change_policy::NNPolicy
    intersect_policy::NNPolicy
    action_map::Vector{LatLonAccel}
end

function decompose(s::Scene)
    lc_mdp = DrivingMDP()
    rt_mdp = DrivingIntersectMDP()
    ego = s[findfirst(1, s)]
    def = VehicleDef()
    s_lc = Scene()
    s_rt = Scene()
    lc_road = gen_straight_roadway(2, road_length)
    rt_road = gen_simple_intersection()

    lc_state2 = VehicleState(Frenet(lc_road[LaneTag(1,2)], 0.0), lc_road, 10.0)
    lc_veh2 = Entity(lc_state2, def, 2)
    lc_state3 = VehicleState(Frenet(lc_road[LaneTag(1,2)], 10.0), lc_road, 10.0)
    lc_veh3 = Entity(lc_state3, def, 3)

    B = VecSE2(0.0,0.0,0.0)
    rt_state2 = VehicleState(B + polar(50.0,-π), rt_road, 10.0)
    rt_veh2 = Vehicle(rt_state2, def, 2)
    rt_state3 = VehicleState(B + polar(30.0,-π), rt_road, 10.0)
    rt_veh3 = Vehicle(rt_state2, def, 3)
    lc_position = (ego.state.posF.s/113.)*100.
    if ego.state.posF.roadind.tag == LaneTag(3,1)
        lane_lc = LaneTag(1,2)
        lane_rt = LaneTag(2,1)
    elseif ego.state.posF.roadind.tag == LaneTag(2,1)
        lane_lc = LaneTag(1,1)
        lane_rt = LaneTag(2,1)
    elseif ego.state.posF.roadind.tag == LaneTag(1,1)
        lane_lc = LaneTag(1,1)
        lane_rt = LaneTag(1,1)
    else
        lane_lc = LaneTag(1,1)
        lane_rt = LaneTag(1,1)
    end
    lc_state1 = VehicleState(Frenet(lc_road[lane_lc], lc_position), lc_road, 10.0)
    lc_veh1 = Vehicle(lc_state1, def, 1)
    rt_state1 = VehicleState(Frenet(lc_road[lane_lc], ego.state.posF.s), rt_road, 10.0)
    rt_veh1 = Vehicle(rt_state1, def, 1)
    push!(s_lc, lc_veh1)
    push!(s_lc, lc_veh2)
    push!(s_lc, lc_veh3)
    push!(s_rt, rt_veh1)
    push!(s_rt, rt_veh2)
    push!(s_rt, rt_veh3)
    return s_lc, s_rt
end

function POMDPPolicies.actionvalues(p::ComposedPolicy, s::Scene)
    s_lc, s_rt = decompose(s)
    return actionvalues(p.lane_change_policy, s_lc) .+ actionvalues(p.intersect_policy, s_in)
end

#decompose - can normalize, change length to 113

function POMDPPolicies.actionvalues(p::ComposedPolicy, v::AbstractVector)
    mdp = DrivingCombinedMDP()
    scene = POMDPs.convert_s(v, mdp)
    return POMDPs.actionvalues(p, scene)
end

# function POMDPPolicies.actionvalues(p::ComposedPolicy, v::V) where V<:AbstractArray
#     s_lc = v[1]
#     s_in = v[2]
#     return actionvalues(p.lane_change_policy, s_lc) .+ actionvalues(p.intersect_policy, s_in)
# end

function POMDPPolicies.action(p::ComposedPolicy, s::Scene)
    vals = actionvalues(p, s)
    imax = argmax(vals)
    mdp = DrivingCombinedMDP()
    action_map = actions(mdp)
    return action_map[imax]
end
