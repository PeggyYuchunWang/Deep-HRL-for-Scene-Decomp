mdp = DrivingMDP()
model = Chain(Dense(12, 32, tanh), Dense(32, 32, tanh), Dense(32, n_actions(mdp)))

solver = DeepQLearningSolver(qnetwork = model, max_steps=100_000,
                             learning_rate=0.001,log_freq=500,
                             recurrence=false,double_q=true, dueling=false, prioritized_replay=true, eps_end=0.01,
                             target_update_freq = 3000, eps_fraction=0.5, train_start=10000, buffer_size=400000,
                             eval_freq=10_000, exploration=my_exploration)
policy = solve(solver, mdp)

hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, policy, POMDPs.initialstate(mdp, MersenneTwister(1)));

@manipulate for frame_index in 1 : n_steps(history)
    render(history.state_hist[frame_index], mdp.roadway, cam=FitToContentCamera(), car_colors=carcolors)
end
