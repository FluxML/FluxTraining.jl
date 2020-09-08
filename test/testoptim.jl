include("./imports.jl")

@testset ExtendedTestSet "Optimizer and OptimParams" begin
    opt = ADAM(0.01)
    @test getoptimparam(opt, LR) == 0.01
    setoptimparam!(opt, LR, 0.001)
    @test getoptimparam(opt, LR) == 0.001
end
