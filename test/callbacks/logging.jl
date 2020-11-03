include("../imports.jl")

tbbackend() = TensorBoardBackend(mktempdir())

@testset ExtendedTestSet "`LogMetrics`" begin
    cb = LogMetrics(tbbackend())
    learner = testlearner(Metrics(accuracy), Recorder(), cb)
    @test_nowarn fit!(learner, 1)
end


@testset ExtendedTestSet "`LogHyperParams`" begin
    cb = LogHyperParams(tbbackend())
    learner = testlearner(Recorder(), Scheduler(), cb)
    @test_nowarn fit!(learner, 1)
end

@testset ExtendedTestSet "`LogHistograms`" begin
    cb = LogHistograms(tbbackend(), freq = 5)
    learner = testlearner(Recorder(), Scheduler(), cb)
    @test_nowarn fit!(learner, 5)
end
