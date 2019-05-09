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

@load "weights/simple_lanechange_policy_weights_test6.jld2" weights
simple_lc_policy = NNPolicy(lc_mdp, weights, actions(lc_mdp), 1)
@load "weights/simple_intersection_policy_weights_test6.jld2" weights
simple_in_policy = NNPolicy(in_mdp, weights, actions(in_mdp), 1)

s0 = POMDPs.initialstate(mdp, MersenneTwister(1))
s_lc, s_rt = decompose(s0)

vals = actionvalues(simple_lc_policy, s_lc) .+ actionvalues(simple_in_policy, s_rt)
compPolicy = ComposedPolicy(simple_lc_policy, simple_in_policy, actions(mdp))

policy1 = RandomPolicy(mdp)
# policy1 = FunctionPolicy(s -> actions(mdp)[LatLonAccel(0.0, 0.0)])
hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, compPolicy, POMDPs.initialstate(mdp, MersenneTwister(1)));

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

w = Window() # this should open a window
ui = @manipulate for frame_index = 1: n_steps(history) + 1
    d = distance(history.state_hist[frame_index], mdp)
    string_d = string("distance: ", d)
    text_overlay = TextOverlay(text=[string_d], font_size=30, pos = VecE2(50.0, 100.0))
    AutoViz.render(history.state_hist[frame_index], mdp.roadway, [text_overlay], cam=FitToContentCamera(), car_colors=carcolors)
end
body!(w, ui) # send the widget in the window and you can interact with it

@show undiscounted_reward(history)

@show reachgoal(history.state_hist[n_steps(history)], mdp.goal_pos)
@show reachgoal(history.state_hist[end], mdp.goal_pos)
@show POMDPs.isterminal(mdp, history.state_hist[end])
@show n_steps(history)
@show POMDPs.reward(mdp, history.state_hist[n_steps(history)], LatLonAccel(0.0, 0.0), history.state_hist[n_steps(history)])
@show POMDPs.reward(mdp, history.state_hist[end], LatLonAccel(0.0, 0.0), history.state_hist[end])

# @save "policies/composition_intersection_policy_decomp.jld2" compPolicy
# @load "policies/composition_intersection_policy_decomp.jld2" compPolicy
