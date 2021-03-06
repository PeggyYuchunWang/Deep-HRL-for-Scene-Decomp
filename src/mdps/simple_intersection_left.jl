include("../AutomotiveHRLSceneDecomp.jl")
include("../utils/helpers.jl")

# state = ego vehicle state, action = tuple(long acceleration, steering)
@with_kw struct DrivingIntersectLeftMDP <: MDP{Scene, LatLonAccel} # MDP{State, Action}
    r_goal::Float64 = 1.0 # reward for reaching goal (default 1)
    discount_factor::Float64 = 0.9 # discount
    cost::Float64 = -1.0
    road_length::Float64 = 113.0
    roadway::Roadway = gen_simple_intersection_left()
    delta_t::Float64 = 0.5
    ego_id::Int64 = 1
    n_cars::Int64 = 3
    models::Dict{Int, DriverModel} = Dict()
    goal_pos::Frenet = get_end_frenet(roadway, LaneTag(1,1))
    speed_limit::Float64 = 15.0
end

const LAT_LON_ACTIONS = [LatLonAccel(y, x) for x in -4:1.0:3 for y in -1:0.1:1]

function POMDPs.actions(mdp::DrivingIntersectLeftMDP)
    return LAT_LON_ACTIONS
end

POMDPs.n_actions(mdp::DrivingIntersectLeftMDP) = length(LAT_LON_ACTIONS)

function POMDPs.initialstate(mdp::DrivingIntersectLeftMDP, rng::AbstractRNG)
    scene = Scene()
    def = VehicleDef()
    state1 = VehicleState(Frenet(mdp.roadway[LaneTag(1,1)],15.0), mdp.roadway, 10.0)
    veh1 = Vehicle(state1, def, 1)

    mdp.models[1] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
    mdp.models[2] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
    mdp.models[3] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))

    A = VecSE2(0.0,DEFAULT_LANE_WIDTH,-π)

    state2 = VehicleState(A - polar(50.0,-π), mdp.roadway, 10.0)
    veh2 = Vehicle(state2, def, 2)

    state3 = VehicleState(A - polar(30.0,-π), mdp.roadway, 10.0)
    veh3 = Vehicle(state3, def, 3)

    push!(scene, veh1)
    push!(scene, veh2)
    push!(scene, veh3)
    return scene
end

function POMDPs.generate_s(mdp::DrivingIntersectLeftMDP, s::Scene, a::LatLonAccel, rng::AbstractRNG)
    sp = deepcopy(s)
    mdp.models[mdp.ego_id].a = a
    actions = Vector{LatLonAccel}(undef, mdp.n_cars)
    get_actions!(actions, s, mdp.roadway, mdp.models)
    ego = sp[findfirst(mdp.ego_id, s)]
    tick!(sp, mdp.roadway, actions, mdp.delta_t)
    return sp
end

function POMDPs.discount(mdp::DrivingIntersectLeftMDP)
    return mdp.discount_factor
end

function POMDPs.convert_s(tv::Type{V}, s::Scene, mdp::DrivingIntersectLeftMDP) where V<:AbstractArray
    ego = s[findfirst(mdp.ego_id, s)]
    laneego = ego.state.posF.roadind.tag.lane
    laneego = Flux.onehot(laneego,[1,2])
    other_vehicles = []
    for veh in s
        if veh.id != mdp.ego_id
            push!(other_vehicles, veh.state)
        end
    end
    svec = Float64[ego.state.posF.s/mdp.road_length, ego.state.v/20.0, laneego...]
    for veh in other_vehicles
        push!(svec, veh.posF.s/mdp.road_length)
        push!(svec, veh.v/20.0)
        laneveh = Flux.onehot(veh.posF.roadind.tag.lane,[1,2])
        push!(svec, laneveh...)
    end
    return svec
end

# TODO: change for intersectMDP
function POMDPs.convert_s(ts::Type{Scene}, v::V, mdp::DrivingIntersectLeftMDP) where V<:AbstractArray
    scene = Scene()
    def = VehicleDef()

    lane1 = v[3] == 1 ? LaneTag(1,1) : LaneTag(1,2)
    state1 = VehicleState(Frenet(mdp.roadway[lane1], v[1]*mdp.road_length), mdp.roadway, v[2]*20.0)
    veh1 = Entity(state1, def, mdp.ego_id)


    lane2 = v[7] == 1 ? LaneTag(1,1) : LaneTag(1,2)
    state2 = VehicleState(Frenet(mdp.roadway[lane2], v[5]*mdp.road_length), mdp.roadway, v[6]*20.0)
    veh2 = Entity(state2, def, 2)


    lane3 = v[11] == 1 ? LaneTag(1,1) : LaneTag(1,2)
    state3 = VehicleState(Frenet(mdp.roadway[lane3], v[9]*mdp.road_length), mdp.roadway, v[10]*20.0)
    veh3 = Entity(state3, def, 3)

    push!(scene, veh1)
    push!(scene, veh2)
    push!(scene, veh3)

    return scene
end

function POMDPs.isterminal(mdp::DrivingIntersectLeftMDP, s::Scene)
    ego = s[findfirst(mdp.ego_id, s)]
    if reachgoal(s, mdp.goal_pos) || collision_helper(s, mdp) || off_road(s, mdp)
        return true
    else
        return false
    end
end

function POMDPs.reward(mdp::DrivingIntersectLeftMDP, s::Scene, a::LatLonAccel, sp::Scene)
    ego = s[findfirst(mdp.ego_id, s)]
    if collision_helper(sp, mdp) || off_road(sp, mdp)
        return -1.0
    elseif reachgoal(sp, mdp)
        return 1.0
    else
        r = -0.01*distance(sp, mdp)/mdp.road_length
        if off_road(sp, mdp)
#             r -= -0.01
        end
        return r
    end
end

function POMDPs.actionindex(mdp::DrivingIntersectLeftMDP, a::LatLonAccel)
    return findfirst(isequal(a), POMDPs.actions(mdp))
end
