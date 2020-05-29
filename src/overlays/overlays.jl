include("../src/AutomotiveHRLSceneDecomp.jl")

@with_kw mutable struct DecompWeightsOverlay <: SceneOverlay
    pos::Frenet
    frac::Float64
    color::Colorant

    color = get(ColorSchemes.redgreensplit, frac)
end

function AutoViz.render!(rendermodel::RenderModel, overlay::DecompWeightsOverlay, scene::Scene, roadway::Any)
    width, height = 1, 1 # based on discretization
    color = get(ColorSchemes.redgreensplit, overlay.frac)
    add_instruction!(rendermodel, render_rect,
        (overlay.pos.x, overlay.pos.y, width, height, overlay.color))
    return rendermodel
end
