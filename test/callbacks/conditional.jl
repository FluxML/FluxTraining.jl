include("../imports.jl")


@testset ExtendedTestSet "`throttle` steps" begin
    cb = CustomCallback(Events.BatchBegin, TrainingPhase) do learner
            throw(CancelBatchException("test"))
    end
    # only cancel every second batch
    throttledcb = throttle(cb, Events.BatchBegin, freq = 2)

    learner = testlearner(Recorder(), throttledcb, coeff = 3)
    fit!(learner, TrainingPhase())
    # should have made 8 steps instead of 16
    @test learner.cbstate.history[TrainingPhase()].steps == 8
end
