include("../AutomotiveHRLSceneDecomp.jl")

# TODO: finish this
mutable struct ConstantDriver{A} <: DriverModel{LatLonAccel}
    action::LatLonAccel
end

function ConstantDriver(veh::Entity{VehicleState, VehicleDef, Int64}, a::LatLonAccel, roadway::Roadway, timestep::Float64)
    AutomotiveDrivingModels.propagate(veh, a, roadway, timestep)
end

#observe, rand
