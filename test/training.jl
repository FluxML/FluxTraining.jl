include("./imports.jl")


@testset ExtendedTestSet "Basic `Learner` convergence" begin
    learner = testlearner(coeff = 3)
    fit!(learner, repeat([TrainingPhase()], 5))
    @test  learner.model.coeff[1] ≈ 3 atol = 0.1
end
