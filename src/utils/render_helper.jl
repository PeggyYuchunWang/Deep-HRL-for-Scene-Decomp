include("../src/AutomotiveHRLSceneDecomp.jl")
include("../src/mdps/simple_two_lane.jl")
include("../src/utils/helpers.jl")

@with_kw struct DrivingMDPViz
    s::Scene = Scene()
    colors # whatever you need
    some_text::Union{Nothing, ::String} = nothing
end

function AutoViz.render!(rm::RenderModel, viz::DrivingMDPViz)
    render!(rm, viz.s) # render the scene
    if viz.some_text != nothing
          #render the text
    end
end
