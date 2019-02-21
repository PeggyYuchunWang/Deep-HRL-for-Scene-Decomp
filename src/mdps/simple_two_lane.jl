# state = ego vehicle state, action = tuple(long acceleration, steering)
@with_kw struct DrivingMDP <: MDP{Scene, LatLonAccel} # MDP{State, Action}
    r_goal::Float64 = 1.0 # reward for reaching goal (default 1)
    discount_factor::Float64 = 0.9 # discount
    cost::Float64 = -1.0
    road_length::Float64 = 100.0
    roadway::Roadway = gen_straight_roadway(2, road_length)
    delta_t::Float64 = 1.0
    ego_id::Int64 = 1
    timestep::Float64 = 0.1
    n_cars::Int64 = 3
    carcolors::Dict{Int,Colorant} = Dict()
    models::Dict{Int, DriverModel} = Dict()
    goal_pos::Frenet = Frenet(roadway[LaneTag(1,2)], road_length)
end

function POMDPs.actions(mdp::DrivingMDP)
    return LAT_LON_ACTIONS
end

POMDPs.n_actions(mdp::DrivingMDP) = length(LAT_LON_ACTIONS)

function POMDPs.initialstate(mdp::DrivingMDP, rng::AbstractRNG)
    scene = Scene()
    def = VehicleDef()
    state1 = VehicleState(Frenet(mdp.roadway[LaneTag(1,1)],0.0), mdp.roadway, 10.0)
    veh1 = Entity(state1, def, mdp.ego_id)

    # carcolors = Dict{Int,Colorant}()
    mdp.carcolors[1] = colorant"red"
    mdp.carcolors[2] = colorant"green"
    mdp.carcolors[3] = colorant"green"

    # models = Dict{Int, DriverModel}()
    # double check dummy
    mdp.models[1] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
    # mdp.models[2] = Tim2DDriver(timestep, rec=SceneRecord(1, timestep, ncars))
    # mdp.models[3] = Tim2DDriver(timestep, rec=SceneRecord(1, timestep, ncars))
    mdp.models[2] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
    mdp.models[3] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))

    state2 = VehicleState(Frenet(roadway[LaneTag(1,2)],0.0), mdp.roadway, 0.0)
    veh2 = Entity(state2, def, 2)

    state3 = VehicleState(Frenet(roadway[LaneTag(1,2)],10.0), mdp.roadway, 0.0)
    veh3 = Entity(state3, def, 3)

    push!(scene, veh1)
    push!(scene, veh2)
    push!(scene, veh3)
    return scene
end

function POMDPs.generate_s(mdp::DrivingMDP, s::Scene, a::LatLonAccel, rng::AbstractRNG)
    sp = deepcopy(s)
    mdp.models[mdp.ego_id].a = a
    actions = Vector{LatLonAccel}(undef, mdp.n_cars)
    get_actions!(actions, s, mdp.roadway, mdp.models)
    ego = sp[findfirst(mdp.ego_id, s)]
    tick!(sp, mdp.roadway, actions, mdp.delta_t)
    return sp
end

function POMDPs.discount(mdp::DrivingMDP)
    return mdp.discount_factor
end

function POMDPs.convert_s(::Type{V}, s::Scene, mdp::DrivingMDP) where V<:AbstractArray
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

function POMDPs.isterminal(mdp::DrivingMDP, s::Scene)
    ego = s[findfirst(mdp.ego_id, s)]
    if ego.state.posF.s >= mdp.road_length || collision_helper(s, mdp) || off_road(s, mdp)
        return true
    else
        return false
    end
end

function POMDPs.reward(mdp::DrivingMDP, s::Scene, a::LatLonAccel, sp::Scene)
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

function POMDPs.actionindex(mdp::DrivingMDP, a::LatLonAccel)
    return findfirst(isequal(a), POMDPs.actions(mdp))
end
