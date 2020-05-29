include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/left_turn.jl")
include("../src/utils/helpers.jl")
# using AutomotiveHRLSceneDecomp

mdp = DrivingLeftTurnMDP()
model = Chain(Dense(27, 32, relu), Dense(32, 32, relu), Dense(32, n_actions(mdp)))

solver = DeepQLearningSolver(qnetwork = model, max_steps=100_000,
                             learning_rate=0.001,log_freq=500,
                             recurrence=false,double_q=true, dueling=false, prioritized_replay=true, eps_end=0.01,
                             target_update_freq = 3000, eps_fraction=0.5, train_start=10000, buffer_size=400000,
                             eval_freq=10_000, logdir="log/left_turn_lane_final1/", batch_size=128)
                             # exploration_policy=masked_linear_epsilon_greedy(1_000_000, 0.5, 0.01),

# @load "policies/left_turn_lane_policy.jld2" policy
# policy = solve(solver, mdp)
# weights = getnetwork(policy)
# @save "weights/left_turn_lane_weights_final1.jld2" weights
# @load "weights/left_turn_lane_weights_final1.jld2" weights
# policy = NNPolicy(mdp, weights, actions(mdp), 1)
policy1 = RandomPolicy(mdp)
# policy1 = FunctionPolicy(s -> LatLonAccel(0., 0.))
# @show actions(mdp)
hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, policy1, POMDPs.initialstate(mdp, MersenneTwister(1)));

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

w = Window() # this should open a window
ui = @manipulate for frame_index = 1: n_steps(history)+1
     d = distance(history.state_hist[frame_index], mdp)
     string_d = string("distance: ", d)
     text_overlay = TextOverlay(text=[string_d], font_size=30, pos = VecE2(50.0, 100.0))
     AutoViz.render(history.state_hist[frame_index], mdp.roadway, [text_overlay], cam=FitToContentCamera(), car_colors=carcolors)
end
body!(w, ui) # send the widget in the window and you can interact with it

@show undiscounted_reward(history)

@show reachgoal(history.state_hist[n_steps(history)], mdp)
@show reachgoal(history.state_hist[end], mdp)
@show POMDPs.isterminal(mdp, history.state_hist[end])
@show n_steps(history)
@show POMDPs.reward(mdp, history.state_hist[n_steps(history)], LatLonAccel(0.0, 0.0), history.state_hist[n_steps(history)])
@show POMDPs.reward(mdp, history.state_hist[end], LatLonAccel(0.0, 0.0), history.state_hist[end])

# @save "policies/left_turn_lane_policy.jld2" policy
# @load "policies/left_turn_lane_policy.jld2" policy
