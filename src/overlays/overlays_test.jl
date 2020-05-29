include("../src/AutomotiveHRLSceneDecomp.jl")

road_length = 100.0
roadway = gen_straight_roadway(2, road_length)

decomp_weights_overlay = DecompWeightsOverlay()

AutoViz.render(Scene(), roadway, [decomp_weights_overlay], cam=FitToContentCamera())
