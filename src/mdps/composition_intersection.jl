include("../AutomotiveHRLSceneDecomp.jl")
include("../utils/helpers.jl")

# state = ego vehicle state, action = tuple(long acceleration, steering)
@with_kw struct DrivingCombinedMDP <: MDP{Scene, LatLonAccel} # MDP{State, Action}
    r_goal::Float64 = 1.0 # reward for reaching goal (default 1)
    discount_factor::Float64 = 0.9 # discount
    cost::Float64 = -1.0
    road_length::Float64 = 113.0
    roadway::Roadway = gen_composition_intersection()
    delta_t::Float64 = 0.5
    ego_id::Int64 = 1
    n_cars::Int64 = 3
    models::Dict{Int, DriverModel} = Dict()
    goal_lane::LaneTag = LaneTag(3,1)
    goal_pos::Frenet = get_end_frenet(roadway, goal_lane)
    speed_limit::Float64 = 15.0
    lane_width::Float64 = DEFAULT_LANE_WIDTH
end

# TODO: change to -4, 3
const LAT_LON_ACTIONS = [LatLonAccel(y, x) for x in -4:1.0:3 for y in -1:0.1:1]

function POMDPs.actions(mdp::DrivingCombinedMDP)
    return LAT_LON_ACTIONS
end

POMDPs.n_actions(mdp::DrivingCombinedMDP) = length(LAT_LON_ACTIONS)

function POMDPs.initialstate(mdp::DrivingCombinedMDP, rng::AbstractRNG)
    scene = Scene()
    def = VehicleDef()

    mdp.models[1] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
    mdp.models[2] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
    mdp.models[3] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))

    B = VecSE2(0.0,0.0,0.0)

    state1 = VehicleState(Frenet(mdp.roadway[LaneTag(1,1)],15.0), mdp.roadway, 10.0)
    # state1 = VehicleState(B + polar(20.0,-π), mdp.roadway, 10.0) # test for q-decomp
    veh1 = Vehicle(state1, def, 1)

    state2 = VehicleState(B + polar(50.0,-π), mdp.roadway, 10.0)
    veh2 = Vehicle(state2, def, 2)

    state3 = VehicleState(B + polar(30.0,-π), mdp.roadway, 10.0)
    veh3 = Vehicle(state3, def, 3)

    push!(scene, veh1)
    push!(scene, veh2)
    push!(scene, veh3)
    return scene
end

function POMDPs.generate_s(mdp::DrivingCombinedMDP, s::Scene, a::LatLonAccel, rng::AbstractRNG)
    sp = deepcopy(s)
    mdp.models[mdp.ego_id].a = a
    actions = Vector{LatLonAccel}(undef, mdp.n_cars)
    get_actions!(actions, s, mdp.roadway, mdp.models)
    ego = sp[findfirst(mdp.ego_id, s)]
    tick!(sp, mdp.roadway, actions, mdp.delta_t)
    return sp
end

function POMDPs.discount(mdp::DrivingCombinedMDP)
    return mdp.discount_factor
end

function POMDPs.convert_s(tv::Type{V}, s::Scene, mdp::DrivingCombinedMDP) where V<:AbstractArray
    ego = s[findfirst(mdp.ego_id, s)]
    laneego = ego.state.posF.roadind.tag.segment
    laneego = Flux.onehot(laneego,[1, 2, 3])
    other_vehicles = []
    for veh in s
        if veh.id != mdp.ego_id
            push!(other_vehicles, veh.state)
        end
    end
    svec = Float64[ego.state.posF.s/mdp.road_length, ego.state.posF.t/mdp.lane_width, ego.state.v/mdp.speed_limit, laneego...]
    for veh in other_vehicles
        push!(svec, veh.posF.s/mdp.road_length)
        push!(svec, veh.posF.t/mdp.lane_width)
        push!(svec, veh.v/mdp.speed_limit)
        laneveh = Flux.onehot(veh.posF.roadind.tag.segment, [1,2, 3])
        push!(svec, laneveh...)
    end
    return svec
end

# TODO: change for intersectMDP
function POMDPs.convert_s(ts::Type{Scene}, v::V, mdp::DrivingCombinedMDP) where V<:AbstractArray
    scene = Scene()
    def = VehicleDef()

    lane1 = v[6] == 1 ? LaneTag(3,1) : LaneTag(1,1)
    state1 = VehicleState(Frenet(mdp.roadway[lane1], v[1]*mdp.road_length, v[2]*mdp.lane_width), mdp.roadway, v[3]*mdp.speed_limit)
    veh1 = Entity(state1, def, mdp.ego_id)


    lane2 = v[12] == 1 ? LaneTag(3,1) : LaneTag(1,1)
    state2 = VehicleState(Frenet(mdp.roadway[lane2], v[7]*mdp.road_length, v[8]*mdp.lane_width), mdp.roadway, v[9]*mdp.speed_limit)
    veh2 = Entity(state2, def, 2)


    lane3 = v[18] == 1 ? LaneTag(3,1) : LaneTag(1,1)
    state3 = VehicleState(Frenet(mdp.roadway[lane3], v[13]*mdp.road_length, v[14]*mdp.lane_width), mdp.roadway, v[15]*mdp.speed_limit)
    veh3 = Entity(state3, def, 3)

    push!(scene, veh1)
    push!(scene, veh2)
    push!(scene, veh3)

    return scene
end

function POMDPs.isterminal(mdp::DrivingCombinedMDP, s::Scene)
    ego = s[findfirst(mdp.ego_id, s)]
    if reachgoal(s, mdp.goal_pos) || collision_helper(s, mdp) || off_road(s, mdp)
        return true
    else
        return false
    end
end

function POMDPs.reward(mdp::DrivingCombinedMDP, s::Scene, a::LatLonAccel, sp::Scene)
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

function POMDPs.actionindex(mdp::DrivingCombinedMDP, a::LatLonAccel)
    return findfirst(isequal(a), POMDPs.actions(mdp))
end
