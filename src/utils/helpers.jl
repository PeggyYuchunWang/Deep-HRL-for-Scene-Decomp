include("../AutomotiveHRLSceneDecomp.jl")
# using AutomotiveHRLSceneDecomp

function collision_helper(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    for veh in s
        if veh.id != mdp.ego_id
            if collision_checker(ego, veh)
                return true
            end
        end
    end
    return false
end

function off_road(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    if abs(ego.state.posF.t) >= 1.5
        return true
    end
    return false
end

function distance(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    goal = get_posG(mdp.goal_pos, mdp.roadway)
    d = norm(VecE2(goal - ego.state.posG))
    return d
end

# TODO: make ego state posF.s abs value
function reachgoal(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    if mdp.goal_pos.roadind.tag == ego.state.posF.roadind.tag && ego.state.posF.s >= mdp.road_length && abs(mdp.goal_pos.t-ego.state.posF.t) <= 0.5
        return true
    end
    return false
end

function reachgoal(s::Scene, goal_pos::Frenet)
    ego = s[findfirst(1, s)]
    if goal_pos.roadind.tag == ego.state.posF.roadind.tag && abs(goal_pos.s-ego.state.posF.s) <= 0.5 && abs(goal_pos.t-ego.state.posF.t) <= 0.5
        return true
    end
    return false
end

function safe_actions(mdp::MDP, s::Scene)
    # TODO: avoid hard coded numbers, put in MDP
    max_brake = -1.0
    safe_acts = [LatLonAccel(max_brake, 0.0)]
    for a in actions(mdp)
        sp = deepcopy(s)
        action_list = LatLonAccel[LatLonAccel(0.,0.0) for veh in s if veh.id != mdp.ego_id]
        tick!(sp, mdp.roadway, [a, action_list...], mdp.delta_t)
        ego = sp[findfirst(mdp.ego_id, sp)]
        if !off_road(sp, mdp) && !collision_helper(sp, mdp) && ego.state.v < mdp.speed_limit
            push!(safe_acts, a)
        end
    end
    return safe_acts
end

function safe_actions(mdp::MDP, o::AbstractArray)
    s = POMDPs.convert_s(Scene, o, mdp)
    return safe_actions(mdp, s)
end

function best_action(acts::Vector{A}, val::AbstractArray{T}, problem::M) where {A, T <: Real, M <: Union{POMDP, MDP}}
    best_a = acts[1]
    best_ai = actionindex(problem, best_a)
    best_val = val[best_ai]
    for a in acts
        ai = actionindex(problem, a)
        if val[ai] > best_val
            best_val = val[ai]
            best_ai = ai
            best_a = a
        end
    end
    return best_a::A
end

function masked_linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float64, eps_end::Float64)
    # define function that will be called to select an action in DQN
    # only supports MDP environments
    function action_masked_epsilon_greedy(policy::AbstractNNPolicy, env::MDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
        eps = DeepQLearning.update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        acts = safe_actions(mdp, obs) #XXX using pomdp global variable replace by safe_actions(mask, obs)
        val = actionvalues(policy, obs) #change this
        if rand(rng) < eps
            return (rand(rng, acts), eps)
        else
            return (best_action(acts, val, env.problem), eps)
        end
    end
    return action_masked_epsilon_greedy
end

function action_masked_epsilon_greedy(policy::AbstractNNPolicy, env::MDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
    eps = DeepQLearning.update_epsilon(global_step, eps_fraction, eps_end, max_steps)
    acts = safe_actions(mdp, obs) #XXX using pomdp global variable replace by safe_actions(mask, obs)
    val = actionvalues(policy, obs)
    if rand(rng) < eps
        return (rand(rng, acts), eps)
    else
        return (best_action(acts, val, env.problem), eps)
    end
end

function append_to_curve!(target::Curve, newstuff::Curve)
    s_end = target[end].s
    for c in newstuff
        push!(target, CurvePt(c.pos, c.s+s_end, c.k, c.kd))
    end
    return target
end

function gen_simple_intersection()
    # new roadway
    roadway = Roadway();
    # Define coordinates of the entry and exit points to the intersection
    r = 5.0 # turn radius
    B = VecSE2(0.0,0.0,0.0)
    D = VecSE2(r+DEFAULT_LANE_WIDTH,-r,π/2)
    E = VecSE2(2r+DEFAULT_LANE_WIDTH,0,0)

    # Append right turn coming from below
    curve = gen_straight_curve(convert(VecE2, D+VecE2(0,-50)), convert(VecE2, D), 2)
    append_to_curve!(curve, gen_bezier_curve(D, E, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, E), convert(VecE2, E+VecE2(50,0)), 2))
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    @show length(roadway.segments)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append straight left
    curve = gen_straight_curve(convert(VecE2, B+VecE2(-50,0)), convert(VecE2, B), 2)
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, B), convert(VecE2, E), 2)[2:end])
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))
    return roadway
end

function gen_composition_intersection()
    # new roadway
    roadway = Roadway();
    # Define coordinates of the entry and exit points to the intersection
    r = 5.0 # turn radius
    B = VecSE2(0.0,0.0,0.0)
    D = VecSE2(r+DEFAULT_LANE_WIDTH,-r,π/2)
    E = VecSE2(2r+DEFAULT_LANE_WIDTH,0,0)

    # Append right turn coming from below
    curve = gen_straight_curve(convert(VecE2, D+VecE2(0,-50)), convert(VecE2, D), 2)
    append_to_curve!(curve, gen_bezier_curve(D, E, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, E), convert(VecE2, E+VecE2(50,0)), 2))
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    @show length(roadway.segments)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append straight left
    curve = gen_straight_curve(convert(VecE2, B+VecE2(-50,0)), convert(VecE2, B), 2)
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, B), convert(VecE2, E), 2)[2:end])
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))
    return roadway
end

"""
Returns a Frenet object of the end of the road
Args: Roadway, LaneTag
Returns: Frenet
"""
function get_end_frenet(roadway::Roadway, tag::LaneTag)
    return Frenet(roadway[tag], get_end(roadway[tag]))
end
