include("../AutomotiveHRLSceneDecomp.jl")
# using AutomotiveHRLSceneDecomp

# state = ego vehicle state, action = tuple(long acceleration, steering)
@with_kw mutable struct DrivingStochasticMDP <: MDP{Scene, LatLonAccel} # MDP{State, Action}
    r_goal::Float64 = 1.0 # reward for reaching goal (default 1)
    discount_factor::Float64 = 0.9 # discount
    cost::Float64 = -1.0
    road_length::Float64 = 100.0
    roadway::Roadway = gen_straight_roadway(2, road_length)
    delta_t::Float64 = 0.5
    ego_id::Int64 = 1
    max_cars::Int64 = 6
    n_cars::Int64 = 3
    models::Dict{Int, DriverModel} = Dict()
    goal_lane::LaneTag = LaneTag(1,2)
    goal_pos::Frenet = get_end_frenet(roadway, goal_lane)
    speed_limit::Float64 = 15.0
    lane_width::Float64 = DEFAULT_LANE_WIDTH
end

const LAT_LON_ACTIONS = [LatLonAccel(y, x) for x in -4:1.0:3 for y in -1:0.1:1]

function POMDPs.actions(mdp::DrivingStochasticMDP)
    return LAT_LON_ACTIONS
end

POMDPs.n_actions(mdp::DrivingStochasticMDP) = length(LAT_LON_ACTIONS)

function POMDPs.initialstate(mdp::DrivingStochasticMDP, rng::AbstractRNG)
    scene = Scene()
    def = VehicleDef()
    initial_speed = rand(1:mdp.speed_limit)
    state1 = VehicleState(Frenet(mdp.roadway[LaneTag(1,1)],0.0), mdp.roadway, initial_speed)
    veh1 = Entity(state1, def, mdp.ego_id)
    push!(scene, veh1)

    mdp.n_cars = rand(1:mdp.max_cars) # ego vehicle always first

    mdp.models[1] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
    for i = 2:mdp.max_cars
        if i <= mdp.n_cars
            mdp.models[i] = Tim2DDriver(mdp.delta_t, rec=SceneRecord(1, mdp.delta_t))
            initial_speed = rand(1:mdp.speed_limit)
            initial_position = (i-1)*10.0
            state2 = VehicleState(Frenet(mdp.roadway[LaneTag(1,2)], initial_position), mdp.roadway, initial_speed)
            veh2 = Entity(state2, def, i)
            push!(scene, veh2)
        end
    end
    return scene
end

function POMDPs.generate_s(mdp::DrivingStochasticMDP, s::Scene, a::LatLonAccel, rng::AbstractRNG)
    sp = deepcopy(s)
    mdp.models[mdp.ego_id].a = a
    actions = Vector{LatLonAccel}(undef, mdp.n_cars)
    get_actions!(actions, s, mdp.roadway, mdp.models)
    ego = sp[findfirst(mdp.ego_id, s)]
    tick!(sp, mdp.roadway, actions, mdp.delta_t)
    return sp
end

function POMDPs.discount(mdp::DrivingStochasticMDP)
    return mdp.discount_factor
end

function POMDPs.convert_s(tv::Type{V}, s::Scene, mdp::DrivingStochasticMDP) where V<:AbstractArray
    ego = s[findfirst(mdp.ego_id, s)]
    laneego = ego.state.posF.roadind.tag.lane
    laneego = Flux.onehot(laneego,[1,2])
    other_vehicles = []
    svec = Float64[ego.state.posF.s/mdp.road_length, ego.state.posF.t/mdp.lane_width, ego.state.v/mdp.speed_limit, laneego...]
    for veh in s
        if veh.id != mdp.ego_id
            push!(other_vehicles, veh.state)
        end
    end
    for veh in other_vehicles
        push!(svec, veh.posF.s/mdp.road_length)
        push!(svec, veh.posF.t/mdp.lane_width)
        push!(svec, veh.v/mdp.speed_limit)
        laneveh = Flux.onehot(veh.posF.roadind.tag.lane,[1,2])
        push!(svec, laneveh...)
    end
    if mdp.n_cars < mdp.max_cars
        offset = mdp.max_cars - mdp.n_cars
        for j = 1:offset
            push!(svec, -1)
            push!(svec, -1)
            push!(svec, -1)
            laneveh = [-1, -1]
            push!(svec, laneveh...)
        end
    end
    return svec
end

# each vec has posF.s, posF.t, state.v, Lane1, Lane2
function POMDPs.convert_s(ts::Type{Scene}, v::V, mdp::DrivingStochasticMDP) where V<:AbstractArray
    scene = Scene()
    def = VehicleDef()
    n_params = 5

    lane1 = v[4] == 1 ? LaneTag(1,1) : LaneTag(1,2)
    state1 = VehicleState(Frenet(mdp.roadway[lane1], v[1]*mdp.road_length, v[2]*mdp.lane_width), mdp.roadway, v[3]*mdp.speed_limit)
    veh1 = Entity(state1, def, mdp.ego_id)

    push!(scene, veh1)

    for i = 2:mdp.max_cars
        if i < mdp.n_cars
            lane2 = v[(i-1)*n_params + 4] == 1 ? LaneTag(1,1) : LaneTag(1,2)
            state2 = VehicleState(Frenet(mdp.roadway[lane2], v[(i-1)*n_params + 1]*mdp.road_length, v[(i-1)*n_params + 2]*mdp.lane_width), mdp.roadway, v[(i-1)*n_params + 3]*mdp.speed_limit)
            veh2 = Entity(state2, def, i)
            push!(scene, veh2)
        end
    end
    return scene
end

function POMDPs.isterminal(mdp::DrivingStochasticMDP, s::Scene)
    ego = s[findfirst(mdp.ego_id, s)]
    if reachgoal(s, mdp.goal_pos) || collision_helper(s, mdp) || off_road(s, mdp) || ego.state.posF.s >= mdp.road_length
        return true
    else
        return false
    end
end

function POMDPs.reward(mdp::DrivingStochasticMDP, s::Scene, a::LatLonAccel, sp::Scene)
    ego = s[findfirst(mdp.ego_id, s)]
    if collision_helper(sp, mdp) || off_road(sp, mdp)
        return -1.0
    elseif reachgoal(sp, mdp)
        return 1.0
    else
        r = -0.01*distance(sp, mdp)/mdp.road_length
        return r
    end
end

function POMDPs.actionindex(mdp::DrivingStochasticMDP, a::LatLonAccel)
    return findfirst(isequal(a), POMDPs.actions(mdp))
end
