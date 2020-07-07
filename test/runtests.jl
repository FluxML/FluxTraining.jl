using Test
using TestSetExtensions
using Flux
using FluxTraining

include("./utils.jl")

@testset ExtendedTestSet "FluxTraining.jl" begin
    include("./testfunctional.jl")
    include("./testoptim.jl")
    include("./testtraining.jl")
end
