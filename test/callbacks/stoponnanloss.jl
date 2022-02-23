include("../imports.jl")


@testset "`StopOnNanLoss`" begin
    learner = testlearner(coeff = NaN, Recorder(), StopOnNaNLoss())
    # Epoch will be cancelled
    @test_throws CancelFittingException fit!(learner, 1)
    @test learner.cbstate.history[TrainingPhase()].epochs == 0
end
