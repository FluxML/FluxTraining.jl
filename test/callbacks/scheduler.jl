include("../imports.jl")


@testset ExtendedTestSet "`Scheduler`" begin
    nbatches = 16
    schedule = Schedule([0, 2], [1e-4, 1e-6], Animations.linear())
    learner = testlearner(coeff = 3, Recorder(), Scheduler(LearningRate => schedule), nbatches = nbatches)

    fit!(learner, TrainingPhase())
    @test isapprox(last(learner.cbstate.hyperparams[:LearningRate])[2], ((1e-4 + 1e-6) / 2), atol = 0.1)

    fit!(learner, TrainingPhase())
    @test isapprox(last(learner.cbstate.hyperparams[:LearningRate])[2], 1e-6, atol =0.1)
end
