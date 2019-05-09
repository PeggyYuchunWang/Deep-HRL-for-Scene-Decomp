include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/simple_two_lane.jl")
include("../src/utils/helpers.jl")

mdp = DrivingMDP()

initial = POMDPs.initialstate(mdp, MersenneTwister(1))
gen_s = POMDPs.generate_s(mdp, initial, LatLonAccel(0., 0.), MersenneTwister(1))
ego = gen_s[findfirst(mdp.ego_id, gen_s)]
egostate = ego.state.posF
@show egolane = egostate.roadind.tag
@show ego_pos_s = egostate.s
@show ego_pos_t = egostate.t
@show conv_s1 = POMDPs.convert_s(AbstractArray, gen_s, mdp)
conv_s2 = POMDPs.convert_s(Scene, conv_s1, mdp)

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"green"
carcolors[3] = colorant"green"

policy = RandomPolicy(mdp)
# policy = FunctionPolicy(s -> LatLonAccel(0., 0.))

hr = HistoryRecorder(max_steps=100)
history = simulate(hr, mdp, policy, POMDPs.initialstate(mdp, MersenneTwister(1)));

# history.state_hist[frame_index]

ego_test = history.state_hist[end][findfirst(mdp.ego_id, history.state_hist[end])]
egostate = ego_test.state.posF
@show egolane = egostate.roadind.tag
@show ego_pos_s = egostate.s
@show ego_pos_t = egostate.t
@show conv_s3 = POMDPs.convert_s(AbstractArray, history.state_hist[end], mdp)
conv_s4 = POMDPs.convert_s(Scene, conv_s3, mdp)

w = Window() # this should open a window
ui = @manipulate for frame_index = 1: n_steps(history)+1
     # d = distance(history.state_hist[frame_index], mdp)
     d = distance(conv_s4, mdp)
     string_d = string("distance: ", d)
     text_overlay = TextOverlay(text=[string_d], font_size=30, pos = VecE2(50.0, 100.0))
     # AutoViz.render(history.state_hist[frame_index], mdp.roadway, [text_overlay], cam=FitToContentCamera(), car_colors=carcolors)
     AutoViz.render(conv_s4, mdp.roadway, [text_overlay], cam=FitToContentCamera(), car_colors=carcolors)
end
body!(w, ui) # send the widget in the window and you can interact with it

global eval_reward = 0.0
for frame_index = 1: n_steps(history) + 1
    global eval_reward += POMDPs.reward(mdp, history.state_hist[frame_index], LatLonAccel(0.0, 0.0), history.state_hist[frame_index])
end
@show eval_reward

@show reachgoal(history.state_hist[n_steps(history)], mdp.goal_pos)
@show reachgoal(history.state_hist[end], mdp.goal_pos)
@show POMDPs.isterminal(mdp, history.state_hist[end])
@show n_steps(history)
@show POMDPs.reward(mdp, history.state_hist[n_steps(history)], LatLonAccel(0.0, 0.0), history.state_hist[n_steps(history)])
@show POMDPs.reward(mdp, history.state_hist[end], LatLonAccel(0.0, 0.0), history.state_hist[end])
