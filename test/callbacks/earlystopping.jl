include("../imports.jl")


@testset ExtendedTestSet "StopEarly" begin
    learner = testlearner(Recorder(), Metrics(), EarlyStopping(3), coeff = 3, opt = Descent(10.))
    fit!(learner, 10)
    @test learner.cbstate.history[TrainingPhase()].epochs < 10
end
