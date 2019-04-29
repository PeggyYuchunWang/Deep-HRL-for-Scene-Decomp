include("../AutomotiveHRLSceneDecomp.jl")
# using AutomotiveHRLSceneDecomp

function collision_helper(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    for veh in s
        if veh.id != mdp.ego_id
            if AutomotiveDrivingModels.collision_checker(ego, veh)
                return true
            end
        end
    end
    return false
end

function off_road(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    if abs(ego.state.posF.t) > DEFAULT_LANE_WIDTH/2
        return true
    # elseif ego.state.posF.s >= mdp.road_length
    #     return true
    end
    return false
end

# function off_lane(s::Scene, mdp::MDP)
#     ego = s[findfirst(mdp.ego_id, s)]
#     if ego.state.posF.roadind.tag != LaneTag(4, 1)
#         return true
#     end
#     return false
# end

function distance(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    goal = mdp.goal_pos
    road_lane = mdp.roadway[goal.roadind.tag]
    ego_proj = proj(ego.state.posG, road_lane, mdp.roadway)
    ego_proj = Frenet(ego_proj, mdp.roadway)
    d = abs(goal.s - ego_proj.s)
    # d = norm(VecE2(goal - ego.state.posG))
    return d
end

# TODO: make ego state posF.s abs value, test this, projection
function reachgoal(s::Scene, mdp::MDP)
    ego = s[findfirst(mdp.ego_id, s)]
    goal_lane = mdp.roadway[mdp.goal_pos.roadind.tag]
    ego_proj = proj(ego.state.posG, goal_lane, mdp.roadway)
    ego_proj = Frenet(ego_proj, mdp.roadway)
    if abs(mdp.goal_pos.s-ego_proj.s) <= 1.0 && abs(mdp.goal_pos.t-ego_proj.t) <= DEFAULT_LANE_WIDTH/2-.25
        return true
    end
    return false
end

function reachgoal(s::Scene, goal_pos::Frenet)
    ego = s[findfirst(1, s)]
    goal_lane = mdp.roadway[goal_pos.roadind.tag]
    ego_proj = proj(ego.state.posG, goal_lane, mdp.roadway)
    ego_proj = Frenet(ego_proj, mdp.roadway)
    if abs(goal_pos.s-ego_proj.s) <= 1.0 && abs(goal_pos.t-ego_proj.t) <= DEFAULT_LANE_WIDTH/2-.25
        return true
    end
    return false
end

function reachgoal_posG(s::Scene, mdp::MDP)
    ego = s[findfirst(1, s)]
    goal_pos = get_posG(mdp.goal_pos, mdp.roadway)
    if norm(VecE2(ego.state.posG - goal_pos)) <= DEFAULT_LANE_WIDTH/2-.25
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

function gen_simple_intersection_left()
    # new roadway
    roadway = Roadway();
    # Define coordinates of the entry and exit points to the intersection
    r = 5.0 # turn radius
    A = VecSE2(0.0,DEFAULT_LANE_WIDTH,-π)
    B = VecSE2(0.0,0.0,0.0)
    C = VecSE2(r,-r,-π/2)
    D = VecSE2(r+DEFAULT_LANE_WIDTH,-r,π/2)
    E = VecSE2(2r+DEFAULT_LANE_WIDTH,0,0)
    F = VecSE2(2r+DEFAULT_LANE_WIDTH,DEFAULT_LANE_WIDTH,-π)

    # Append left turn coming from below
    curve = gen_straight_curve(convert(VecE2, D+VecE2(0,-50)), convert(VecE2, D), 2)
    append_to_curve!(curve, gen_bezier_curve(D, A, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, A), convert(VecE2, A+VecE2(-50,0)), 2))
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append straight right
    curve = gen_straight_curve(convert(VecE2, F+VecE2(50,0)), convert(VecE2, F), 2)
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, F), convert(VecE2, A), 2)[2:end])
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))
    return roadway
end

function gen_composition_intersection()
    roadway = Roadway();
    # Define coordinates of the entry and exit points to the intersection
    r = 5.0 # turn radius
    B = VecSE2(0.0,0.0,0.0)
    C = B+VecE2(-50,DEFAULT_LANE_WIDTH)
    D = VecSE2(r+DEFAULT_LANE_WIDTH,-r,π/2)
    E = VecSE2(2r+DEFAULT_LANE_WIDTH,0,0)
    F = B+VecE2(63,DEFAULT_LANE_WIDTH)

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

    # Append second lane
    curve = gen_straight_curve(convert(VecE2, C), convert(VecE2, F), 2)
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))
    return roadway
end

function gen_composition_intersection_second()
    roadway = Roadway();
    # Define coordinates of the entry and exit points to the intersection
    r = 5.0 # turn radius
    B = VecSE2(0.0,0.0,0.0)
    C = B+VecE2(-50,DEFAULT_LANE_WIDTH)
    D = VecSE2(r+DEFAULT_LANE_WIDTH,-r,π/2)
    E = VecSE2(2r+DEFAULT_LANE_WIDTH,0,0)
    F = B+VecE2(63,DEFAULT_LANE_WIDTH)

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

    # Append second lane
    curve = gen_straight_curve(convert(VecE2, C), convert(VecE2, F), 2)
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))
    return roadway
end

function gen_left_turn()
    # new roadway
    roadway = Roadway();

    # Define coordinates of the entry and exit points to the intersection
    r = 5.0 # turn radius
    A = VecSE2(0.0,DEFAULT_LANE_WIDTH,-π)
    B = VecSE2(0.0,0.0,0.0)
    C = VecSE2(r,-r,-π/2)
    D = VecSE2(r+DEFAULT_LANE_WIDTH,-r,π/2)
    E = VecSE2(2r+DEFAULT_LANE_WIDTH,0,0)
    F = VecSE2(2r+DEFAULT_LANE_WIDTH,DEFAULT_LANE_WIDTH,-π)

    # Append straight left
    curve = gen_straight_curve(convert(VecE2, B+VecE2(-100,0)), convert(VecE2, E+VecE2(50,0)), 2)
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append straight right
    curve = gen_straight_curve(convert(VecE2, F+VecE2(50,0)), convert(VecE2, A+VecE2(-100,0)), 2)
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append left turn coming from below
    curve = gen_straight_curve(convert(VecE2, D+VecE2(0,-50)), convert(VecE2, D), 2)
    append_to_curve!(curve, gen_bezier_curve(D, A, 0.9r, 0.9r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, A), convert(VecE2, A+VecE2(-100,0)), 2))
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append right turn coming from below
    curve = gen_straight_curve(convert(VecE2, D+VecE2(0,-50)), convert(VecE2, D), 2)
    append_to_curve!(curve, gen_bezier_curve(D, E, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, E), convert(VecE2, E+VecE2(50,0)), 2))
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append right turn coming from the left
    curve = gen_straight_curve(convert(VecE2, B+VecE2(-100,0)), convert(VecE2, B), 2)
    append_to_curve!(curve, gen_bezier_curve(B, C, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, C), convert(VecE2, C+VecE2(0,-50.0)), 2))
    lane = Lane(LaneTag(length(roadway.segments)+1, 1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Append left turn coming from the right
    curve = gen_straight_curve(convert(VecE2, F+VecE2(50,0)), convert(VecE2, F), 2)
    append_to_curve!(curve, gen_bezier_curve(F, C, 0.9r, 0.9r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, C), convert(VecE2, C+VecE2(0,-50)), 2))
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

struct LaneOverlay <: SceneOverlay
    lane::Lane
    color::Colorant
end

function AutoViz.render!(rendermodel::RenderModel, overlay::LaneOverlay, scene::Scene, roadway::Roadway)
    render!(rendermodel, overlay.lane, roadway, color_asphalt=overlay.color) # this display a lane with the specified color
    return rendermodel
end
