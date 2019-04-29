using POMDPs
using POMDPModels
using DeepQLearning
using Flux

mdp = SimpleGridWorld()
model = Chain(Dense(2, 32, relu), Dense(32,4))
solver = DeepQLearningSolver(qnetwork=model, logdir="yikes/")
a = solve(solver, mdp)
@show a
weights = getnetwork(a)
@show weights
