function collision_helper(s::Scene, mdp::DrivingMDP)
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

function off_road(s::Scene, mdp::DrivingMDP)
    ego = s[findfirst(mdp.ego_id, s)]
    if abs(ego.state.posF.t) >= 1.5
        return true
    end
    return false
end

function distance(s::Scene, mdp::DrivingMDP)
    ego = s[findfirst(mdp.ego_id, s)]
    goal = get_posG(mdp.goal_pos, mdp.roadway)
    d = norm(VecE2(goal - ego.state.posG))
    return d
end

function reachgoal(s::Scene, mdp::DrivingMDP)
    ego = s[findfirst(mdp.ego_id, s)]
    if mdp.goal_pos.roadind.tag == ego.state.posF.roadind.tag && ego.state.posF.s >= mdp.road_length && abs(mdp.goal_pos.t-ego.state.posF.t) <= 0.5
        return true
    end
    return false
end

function safe_actions(mdp::DrivingMDP, s::Scene)
    #check off road, immediate collisions (assume other car constant speed or acc), speed limit
    #epsilon-greedy on actions
    return actions
end
