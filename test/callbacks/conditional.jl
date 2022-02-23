include("../imports.jl")


@testset "`throttle` steps" begin
    cb = CustomCallback(Events.StepBegin, TrainingPhase) do learner
            throw(CancelStepException("test"))
    end
    # only cancel every second batch
    throttledcb = throttle(cb, Events.StepBegin, freq = 2)

    learner = testlearner(Recorder(), throttledcb, coeff = 3)
    epoch!(learner, TrainingPhase())
    # should have made 8 steps instead of 16
    @test learner.cbstate.history[TrainingPhase()].steps == 8
end
