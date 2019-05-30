include("../AutomotiveHRLSceneDecomp.jl")
include("../utils/helpers.jl")

env = UrbanEnv(params = UrbanParams(crosswalk_pos = []))
roadway = car_roadway(env)

scene = Scene()
timestep = 0.1
ncars = 3

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

def = VehicleDef()

models = Dict{Int, DriverModel}()
models[1] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
models[2] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))
models[3] = AutomotivePOMDPs.EgoDriver(LatLonAccel(0.0, 0.0))

state1 = VehicleState(Frenet(roadway[LaneTag(13,1)],0.0), roadway, 10.0)
veh1 = Vehicle(state1, def, 1)
@show veh1.state.posF.roadind.tag

state2 = VehicleState(Frenet(roadway[LaneTag(1,2)],0.0), roadway, 10.0)
veh2 = Vehicle(state2, def, 2)
@show veh2.state.posF.roadind.tag

state3 = VehicleState(Frenet(roadway[LaneTag(1,2)],10.0), roadway, 10.0)
veh3 = Vehicle(state3, def, 3)
@show veh3.state.posF.roadind.tag

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

goal_pos = Frenet(roadway[LaneTag(13,1)], get_end(roadway[LaneTag(13,1)]))
@show get_end(roadway[LaneTag(13,1)])
# goal_pos = get_end_frenet(roadway, LaneTag(3,1))
@show goal_pos
@show scene[1]
@show veh1.state.posF
@show veh2.state.posF.roadind.tag

@show reachgoal(scene, goal_pos)
