include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/simple_two_lane.jl")
include("../src/utils/helpers.jl")

roadway = gen_straight_roadway(2, 100.0)
ego_id = 1

scene = Scene()
def = VehicleDef()
state1 = VehicleState(Frenet(roadway[LaneTag(1,1)],0.0), roadway, 10.0)
veh1 = Entity(state1, def, ego_id)

models[1] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
models[2] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
models[3] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))

state2 = VehicleState(Frenet(roadway[LaneTag(1,2)],0.0), roadway, 0.0)
veh2 = Entity(state2, def, 2)

state3 = VehicleState(Frenet(roadway[LaneTag(1,2)],10.0), roadway, 0.0)
veh3 = Entity(state3, def, 3)

push!(scene, veh1)
push!(scene, veh2)
push!(scene, veh3)

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

AutoViz.render(scene, roadway, cam=FitToContentCamera(), car_colors=carcolors)
