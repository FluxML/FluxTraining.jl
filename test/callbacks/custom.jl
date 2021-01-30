include("../imports.jl")


@testset ExtendedTestSet "`CustomCallback`" begin
    cb = CustomCallback(Events.BatchEnd, TrainingPhase) do learner
            throw(CancelFittingException("test"))
    end
    learner = testlearner(Recorder(), cb, coeff = 3)
    fit!(learner, TrainingPhase())
    @test learner.cbstate.history[TrainingPhase()].epochs == 0
    @test learner.cbstate.history[TrainingPhase()].steps == 0
end
