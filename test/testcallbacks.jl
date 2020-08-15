include("./imports.jl")


@testset ExtendedTestSet "ProgressBarLogger" begin
    learner = dummylearner(1, callbacks = [ProgressBarLogger()])
    @test_nowarn fit!(learner, 1)
end


@testset ExtendedTestSet "PrintMetrics" begin
    learner = dummylearner(1, callbacks = [PrintMetrics()])
    @test_nowarn fit!(learner, 1)
end


@testset ExtendedTestSet "StopOnNanLoss" begin
    # is used by default
    learner = dummylearner(NaN)
    # Epoch will be cancelled
    @test_nowarn fit!(learner, 1)
    @test learner.state.history.epochs == 0
end


@testset ExtendedTestSet "CustomCallback" begin
    cb = CustomCallback{EpochEnd,TrainingPhase}((learner) -> CancelFittingException("test"))
    learner = dummylearner(3, callbacks = [cb])
    fit!(learner, TrainingPhase())
    @test learner.state.history.epochs == 1
end
