include("./imports.jl")

@testset ExtendedTestSet "FluxTraining.jl" begin
    include("./testfunctional.jl")
    include("./testoptim.jl")
    include("./testprotected.jl")
    include("./testtraining.jl")
    include("./testcallbacks.jl")
end
