include("AutomotiveHRLSceneDecomp.jl")

mdp = DrivingMDP()
model = Chain(Dense(12, 32, tanh), Dense(32, 32, tanh), Dense(32, n_actions(mdp)))

solver = DeepQLearningSolver(qnetwork = model, max_steps=300_000,
                             learning_rate=0.001,log_freq=500,
                             recurrence=false,double_q=true, dueling=false, prioritized_replay=true, eps_end=0.01,
                             target_update_freq = 3000, eps_fraction=0.5, train_start=10000, buffer_size=400000,
                             eval_freq=10_000)
    # exploration_policy=masked_linear_epsilon_greedy(300_000, 0.5, 0.01))
policy = solve(solver, mdp)

policy1 = FunctionPolicy(s -> actions(mdp)[2])
hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, policy, POMDPs.initialstate(mdp, MersenneTwister(1)));

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

@manipulate for frame_index in 1 : n_steps(history)
    AutoViz.render(history.state_hist[frame_index], mdp.roadway, cam=FitToContentCamera(), car_colors=carcolors)
end
