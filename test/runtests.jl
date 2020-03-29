using Test
using TestSetExtensions
using Flux
using FluxTraining

include("./utils.jl")

@testset ExtendedTestSet "FluxTraining.jl" begin
    @includetests ARGS
end
