include("../imports.jl")


@testset ExtendedTestSet "StopEarly" begin
    learner = testlearner(Recorder(), Metrics(), EarlyStopping(2), coeff = 3, opt = Descent(10.))
    fit!(learner, 100)
    @test learner.cbstate.history[TrainingPhase()].epochs < 100
end
