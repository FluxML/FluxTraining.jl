include("./imports.jl")


@testset ExtendedTestSet "ProgressBarLogger" begin
    learner = testlearner(coeff = 1, callbacks = [ProgressBarLogger()])
    @test_nowarn fit!(learner, 1)
end


#= Need to suppress stdout
@testset ExtendedTestSet "MetricsLogger" begin
    learner = testlearner(coeff = 1, callbacks = [MetricsLogger()])
    @test_nowarn fit!(learner, 1)
end
=#


@testset ExtendedTestSet "StopOnNanLoss" begin
    # is used by default
    learner = testlearner(coeff = NaN, callbacks = [StopOnNaNLoss()])
    # Epoch will be cancelled
    @test_nowarn fit!(learner, 1)
    @test learner.state.history.epochs == 0
end


@testset ExtendedTestSet "CustomCallback" begin
    cb = CustomCallback{EpochEnd,TrainingPhase}((learner) -> CancelFittingException("test"))
    learner = testlearner(coeff = 3, callbacks = [cb])
    fit!(learner, TrainingPhase())
    @test learner.state.history.epochs == 1
end
