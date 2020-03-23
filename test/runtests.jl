using Test
using TestSetExtensions
using Flux
using Training

include("./utils.jl")

@testset ExtendedTestSet "Training.jl tests" begin
    @includetests ARGS
end
