include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/simple_two_lane_stochastic.jl")
include("../src/utils/helpers.jl")
# using AutomotiveHRLSceneDecomp

mdp = DrivingStochasticMDP()
model = Chain(Dense(30, 64, relu), Dense(64, 64, relu), Dense(64, n_actions(mdp)))

solver = DeepQLearningSolver(qnetwork = model, max_steps=300_000,
                             learning_rate=0.001,log_freq=500,
                             recurrence=false, double_q=true, dueling=false, prioritized_replay=true, eps_end=0.01,
                             target_update_freq = 3000, eps_fraction=0.5, train_start=10_000, buffer_size=400000,
                             eval_freq=10_000,
                             # exploration_policy=masked_linear_epsilon_greedy(1_000_000, 0.5, 0.01),
                             # evaluation_policy=masked_linear_epsilon_greedy(1_000_000, 0., 0.),
                             logdir="log/simple_lane_stochastic1/", batch_size=128)
# @show policy = solve(solver, mdp)
# policy = RandomPolicy(mdp)
# @show weights = getnetwork(policy)

# @save "weights/simple_lanechange_policy_weights_stochastic1.jld2" weights
# @load "weights/simple_lanechange_policy_weights_stochastic1.jld2" weights

@load "weights/simple_lanechange_policy_weights_stochastic1.jld2" weights
policy = NNPolicy(mdp, weights, actions(mdp), 1)

policy1 = FunctionPolicy(s -> LatLonAccel(0., 0.))
# policy1 = RandomPolicy(mdp)


@load "weights/simple_lanechange_policy_weights_stochastic1.jld2" weights
hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, policy1, POMDPs.initialstate(mdp, MersenneTwister(3)));

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"
carcolors[4] = colorant"green"
carcolors[5] = colorant"green"
carcolors[6] = colorant"green"

w = Window() # this should open a window
ui = @manipulate for frame_index = 1: n_steps(history)+1
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

@save "weights/simple_lanechange_policy_weights_stochastic1.jld2" weights
@load "weights/simple_lanechange_policy_weights_stochastic1.jld2" weights

# @save "policies/simple_lanechange_policy_next.jld2" policy
# @load "policies/simple_lanechange_policy_next.jld2" policy
