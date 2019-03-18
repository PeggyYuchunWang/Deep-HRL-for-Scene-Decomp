include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/composition_intersection.jl")
include("../src/mdps/simple_intersection.jl")
include("../src/mdps/simple_two_lane.jl")
include("../src/utils/helpers.jl")
include("../src/utils/value_decomp.jl")
# using AutomotiveHRLSceneDecomp

mdp = DrivingCombinedMDP()
lc_mdp = DrivingMDP()
in_mdp = DrivingIntersectMDP()
simple_lc_policy = RandomPolicy(lc_mdp)
simple_in_policy = RandomPolicy(in_mdp)
@load "policies/simple_lanechange_policy.jld2" policy
simple_lc_policy = policy
@load "policies/simple_intersection_policy.jld2" policy
simple_in_policy = policy

@show simple_lc_policy
@show getnetwork(simple_lc_policy)

s0 = POMDPs.initialstate(mdp, MersenneTwister(1))
s_lc, s_rt = decompose(s0)

@show vals = actionvalues(simple_lc_policy, s_lc) .+ actionvalues(simple_in_policy, s_rt)
compPolicy = ComposedPolicy(simple_lc_policy, simple_in_policy, actions(mdp))
# compPolicy.lane_change_policy =
# compPolicy.intersect_policy =
# compPolicy.action_map =

policy1 = RandomPolicy(mdp)
hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, compPolicy, POMDPs.initialstate(mdp, MersenneTwister(1)));

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

w = Window() # this should open a window
ui = @manipulate for frame_index = 1: n_steps(history)
     AutoViz.render(history.state_hist[frame_index], mdp.roadway, cam=FitToContentCamera(), car_colors=carcolors)
end
body!(w, ui) # send the widget in the window and you can interact with it

reachgoal(history.state_hist[n_steps(history)], mdp.goal_pos)

#@save "composition_intersection_policy_decomp.jld2" policy
#@load "composition_intersection_policy_decomp.jld2" policy
