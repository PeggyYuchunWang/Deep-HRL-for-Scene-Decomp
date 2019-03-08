include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/simple_two_lane.jl")
include("../src/utils/helpers.jl")


roadway = gen_straight_roadway(2, 100.0)
ego_id = 1

scene = Scene()
def = VehicleDef()
state1 = VehicleState(Frenet(roadway[LaneTag(1,1)],0.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)

state2 = VehicleState(Frenet(roadway[LaneTag(1,2)],30.0), roadway, 0.0)
veh2 = Entity(state2, def, 2)

state3 = VehicleState(Frenet(roadway[LaneTag(1,2)],90.0), roadway, 0.0)
veh3 = Entity(state3, def, 3)

push!(scene, veh1)
push!(scene, veh2)
push!(scene, veh3)

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

mdp = DrivingMDP()
@assert collision_helper(scene, mdp) == false

scene = Scene()
state1 = deepcopy(state2)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)
push!(scene, veh2)
push!(scene, veh3)

@assert collision_helper(scene, mdp) == true

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],50.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert off_road(scene, mdp) == false

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],50.0, 20.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert off_road(scene, mdp) == true

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],50.0, 1.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert off_road(scene, mdp) == false

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],50.0, 1.5), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert off_road(scene, mdp) == true

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],50.0, -1.5), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert off_road(scene, mdp) == true

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],50.0, 0.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert distance(scene, mdp) == 50.0

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],100.0, 0.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert distance(scene, mdp) == 0.0

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],0.0, 0.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert distance(scene, mdp) == 100.0

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],20.0, 10.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)
f = Frenet(roadway[LaneTag(1,2)],20.0, 10.0)

@assert distance(scene, mdp) == norm(get_posG(mdp.goal_pos, roadway) - get_posG(f, roadway))

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,2)],20.0, 10.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert reachgoal(scene, mdp) == false

scene = Scene()
state1 = VehicleState(get_posG(mdp.goal_pos, roadway), mdp.goal_pos, 0.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)

@assert reachgoal(scene, mdp) == true

scene = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(1,1)],0.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)
push!(scene, veh1)
push!(scene, veh2)
push!(scene, veh3)

@assert safe_actions(mdp, scene) == actions(mdp)

AutoViz.render(scene, roadway, cam=FitToContentCamera(), car_colors=carcolors)
