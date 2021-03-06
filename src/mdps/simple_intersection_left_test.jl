include("../AutomotiveHRLSceneDecomp.jl")
include("../utils/helpers.jl")

roadway = gen_simple_intersection_left()

scene = Scene()
timestep = 0.1
ncars = 3

A = VecSE2(0.0,DEFAULT_LANE_WIDTH,-π)
B = VecSE2(0.0,0.0,0.0)

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

def = VehicleDef()

models = Dict{Int, DriverModel}()
models[1] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
models[2] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
models[3] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))

state1 = VehicleState(Frenet(roadway[LaneTag(1,1)],15.0), roadway, 10.0)
veh1 = Vehicle(state1, def, 1)
@show veh1.state.posF.roadind.tag

state2 = VehicleState(A - polar(50.0,-π), roadway, 10.0)
veh2 = Vehicle(state2, def, 2)
@show veh2.state.posF.roadind.tag

state3 = VehicleState(A - polar(30.0,-π), roadway, 10.0)
veh3 = Vehicle(state3, def, 3)

push!(scene, veh1)
push!(scene, veh2)
push!(scene, veh3)

nticks = 150
rec = SceneRecord(nticks+1, timestep)
simulate!(rec, scene, roadway, models, nticks)

w = Window() # this should open a window
ui = @manipulate for frame_index in 1 : nframes(rec)
     AutoViz.render(rec[frame_index-nframes(rec)], roadway, cam=FitToContentCamera(), car_colors=carcolors)
end
body!(w, ui) # send the widget in the window and you can interact with it

# goal_pos = Frenet(roadway[LaneTag(1,1)], get_end(roadway[LaneTag(1,1)]))
goal_pos = get_end_frenet(roadway, LaneTag(1,1))

@show reachgoal(rec[0], goal_pos)
