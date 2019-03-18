include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/composition_intersection.jl")
include("../src/utils/helpers.jl")
# using AutomotiveHRLSceneDecomp

mdp = DrivingCombinedMDP()
model = Chain(Dense(12, 32, relu), Dense(32, 32, relu), Dense(32, n_actions(mdp)))

solver = DeepQLearningSolver(qnetwork = model, max_steps=1_000_000,
                             learning_rate=0.001,log_freq=500,
                             recurrence=false,double_q=true, dueling=false, prioritized_replay=true, eps_end=0.01,
                             target_update_freq = 3000, eps_fraction=0.5, train_start=10000, buffer_size=400000,
                             eval_freq=10_000, exploration_policy=masked_linear_epsilon_greedy(1_000_000, 0.5, 0.01),
                             logdir="log/composition_intersection_policy_baseline_2/", batch_size=128)

# @load "policies/composition_intersection_policy_baseline.jld2" policy
policy = solve(solver, mdp)
policy1 = RandomPolicy(mdp)
hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, policy, POMDPs.initialstate(mdp, MersenneTwister(1)));

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

@save "policies/composition_intersection_policy_baseline_2.jld2" policy
@load "policies/composition_intersection_policy_baseline_2.jld2" policy
