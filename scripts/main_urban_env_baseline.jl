include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/urban_env.jl")
include("../src/utils/helpers.jl")
# using AutomotiveHRLSceneDecomp

mdp = DrivingUrbanMDP()
model = Chain(Dense(51, 60, relu), Dense(60, 60, relu), Dense(60, n_actions(mdp)))

solver = DeepQLearningSolver(qnetwork = model, max_steps=300_000,
                             learning_rate=0.001, log_freq=500,
                             recurrence=false,double_q=true, dueling=false, prioritized_replay=true, eps_end=0.1,
                             target_update_freq = 3000, eps_fraction=0.5, train_start=10000, buffer_size=400_000,
                             eval_freq=10_000,
                             # exploration_policy=masked_linear_epsilon_greedy(1_000_000, 0.5, 0.1),
                             logdir="log/urban_env_policy_baseline_final1/", batch_size=128)

# @load "policies/composition_intersection_policy_baseline.jld2" policy
# @load "policies/composition_intersection_policy_baseline_2.jld2" policy
# @load "policies/composition_intersection_policy_baseline_rewardchange.jld2" policy
policy = solve(solver, mdp)
weights = getnetwork(policy)
@save "weights/urban_env_baseline_policy_weights_final1.jld2" weights
@load "weights/urban_env_baseline_policy_weights_final1.jld2" weights
policy = NNPolicy(mdp, weights, actions(mdp), 1)

# policy1 = FunctionPolicy(s -> actions(mdp)[LatLonAccel(0.0, 0.0)])
policy1 = RandomPolicy(mdp)
hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, policy, POMDPs.initialstate(mdp, MersenneTwister(1)));

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

w = Window() # this should open a window
ui = @manipulate for frame_index = 1: n_steps(history) + 1
     AutoViz.render(history.state_hist[frame_index], mdp.roadway, cam=FitToContentCamera(), car_colors=carcolors)
end
body!(w, ui) # send the widget in the window and you can interact with it

@show undiscounted_reward(history)

@show reachgoal(history.state_hist[n_steps(history)], mdp.goal_pos)
@show reachgoal(history.state_hist[end], mdp.goal_pos)
@show POMDPs.isterminal(mdp, history.state_hist[end])
@show n_steps(history)
@show POMDPs.reward(mdp, history.state_hist[n_steps(history)], LatLonAccel(0.0, 0.0), history.state_hist[n_steps(history)])
@show POMDPs.reward(mdp, history.state_hist[end], LatLonAccel(0.0, 0.0), history.state_hist[end])

# @save "policies/composition_intersection_policy_baseline_rewardchange.jld2" policy
# @load "policies/composition_intersection_policy_baseline_rewardchange.jld2" policy
