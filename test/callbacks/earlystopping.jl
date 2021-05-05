include("../imports.jl")


@testset ExtendedTestSet "StopEarly" begin
    learner = testlearner(Recorder(), Metrics(), EarlyStopping(1), ProgressPrinter(), coeff = 3, opt = Descent(0.1))
    fit!(learner, 100)
    @test learner.cbstate.history[TrainingPhase()].epochs < 100
end
