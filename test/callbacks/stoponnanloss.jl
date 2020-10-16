include("../imports.jl")


@testset ExtendedTestSet "`StopOnNanLoss`" begin
    learner = testlearner(coeff = NaN, Recorder(), StopOnNaNLoss())
    # Epoch will be cancelled
    @test_nowarn fit!(learner, 1)
    @test learner.cbstate.history.epochs == 0
end
