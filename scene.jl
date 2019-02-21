using AutomotiveDrivingModels
using AutoViz
using DeepQLearning
using POMDPs
using Flux
using POMDPModels
using POMDPSimulators
using Parameters
using Reel
using Random
using Printf
using POMDPSimulators
using Interact
using AutomotivePOMDPs
using LinearAlgebra
using Revise

scene = Scene()
timestep = 0.1
ncars = 3

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

models = Dict{Int, DriverModel}()
models[1] = Tim2DDriver(timestep, rec=SceneRecord(1, timestep, ncars))
models[2] = Tim2DDriver(timestep, rec=SceneRecord(1, timestep, ncars))
models[3] = Tim2DDriver(timestep, rec=SceneRecord(1, timestep, ncars))

road_length = 100.0 # [meters]
roadway = gen_straight_roadway(2, road_length)
def = VehicleDef()

state1 = VehicleState(Frenet(roadway[LaneTag(1,1)],0.0), roadway, 0.0)
veh1 = Vehicle(state1, def, 1)

state2 = VehicleState(Frenet(roadway[LaneTag(1,2)],0.0), roadway, 0.0)
veh2 = Vehicle(state2, def, 2)

state3 = VehicleState(Frenet(roadway[LaneTag(1,2)],10.0), roadway, 0.0)
veh3 = Vehicle(state3, def, 3)

push!(scene, veh1)
push!(scene, veh2)
push!(scene, veh3)

render(scene, roadway, cam=FitToContentCamera(), car_colors=carcolors)
