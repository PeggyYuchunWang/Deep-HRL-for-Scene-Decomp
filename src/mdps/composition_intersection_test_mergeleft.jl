include("../AutomotiveHRLSceneDecomp.jl")
include("../utils/helpers.jl")

# new roadway
roadway = Roadway();
# Define coordinates of the entry and exit points to the intersection
r = 5.0 # turn radius
A = VecSE2(0.0,DEFAULT_LANE_WIDTH,-π)
B = VecSE2(0.0,0.0,0.0)
# C = VecSE2(r,-r,-π/2)
# D = VecSE2(r+DEFAULT_LANE_WIDTH,-r,π/2)
# E = VecSE2(2r+DEFAULT_LANE_WIDTH,0,0)
F = VecSE2(2r+DEFAULT_LANE_WIDTH,DEFAULT_LANE_WIDTH,-π)
G = B+VecE2(-50,2*DEFAULT_LANE_WIDTH)
H = B+VecE2(63,2*DEFAULT_LANE_WIDTH)

# Append left turn coming from below
curve = gen_straight_curve(convert(VecE2, D+VecE2(0,-50)), convert(VecE2, D), 2)
append_to_curve!(curve, gen_bezier_curve(D, A, 0.6r, 0.6r, 51)[2:end])
append_to_curve!(curve, gen_straight_curve(convert(VecE2, A), convert(VecE2, A+VecE2(-50,0)), 2))
lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

# Append straight right
curve = gen_straight_curve(convert(VecE2, F+VecE2(50,0)), convert(VecE2, F), 2)
append_to_curve!(curve, gen_straight_curve(convert(VecE2, F), convert(VecE2, A), 2)[2:end])
lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

# Append second lane
curve = gen_straight_curve(convert(VecE2, G), convert(VecE2, H), 2)
lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
@show lane.tag
push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

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

state1 = VehicleState(Frenet(roadway[LaneTag(1,1)],0.0), roadway, 10.0)
veh1 = Vehicle(state1, def, 1)

state2 = VehicleState(A - polar(50.0,-π), roadway, 10.0)
veh2 = Vehicle(state2, def, 2)

state3 = VehicleState(A - polar(30.0,-π), roadway, 10.0)
veh3 = Vehicle(state3, def, 3)

push!(scene, veh1)
push!(scene, veh2)
push!(scene, veh3)

scene1 = Scene()
state1 = VehicleState(Frenet(roadway[LaneTag(3,1)], 0.), roadway, 10.0)
veh1 = Vehicle(state1, def, 1)

push!(scene1, veh1)
push!(scene1, veh2)
push!(scene1, veh3)

nticks = 150
rec = SceneRecord(nticks+1, timestep)
simulate!(rec, scene1, roadway, models, nticks)

w = Window() # this should open a window
ui = @manipulate for frame_index in 1 : nframes(rec)
     AutoViz.render(rec[frame_index-nframes(rec)], roadway, cam=FitToContentCamera(), car_colors=carcolors)
end
body!(w, ui) # send the widget in the window and you can interact with it

goal_pos = Frenet(roadway[LaneTag(3,1)], 0.)
@show get_end(roadway[LaneTag(3,1)])
# goal_pos = get_end_frenet(roadway, LaneTag(3,1))
@show goal_pos
@show rec[-150]
@show rec[-150][1].state.posF
@show veh1.state.posF.roadind.tag

# @show reachgoal(rec[0], goal_pos)

@show reachgoal(rec[-150], goal_pos)
