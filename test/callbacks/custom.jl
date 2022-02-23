include("../imports.jl")


@testset "`CustomCallback`" begin
    cb = CustomCallback(Events.StepEnd, TrainingPhase) do learner
        throw(CancelFittingException("test"))
    end
    learner = testlearner(Recorder(), cb, coeff = 3)
    @test_throws CancelFittingException epoch!(learner, TrainingPhase())
    @test learner.cbstate.history[TrainingPhase()].epochs == 0
    @test learner.cbstate.history[TrainingPhase()].steps == 0
end
