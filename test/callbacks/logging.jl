include("../imports.jl")

tbbackend() = TensorBoardBackend(mktempdir())
mlflowbackend() = MLFlowBackend(tracking_uri=ENV["MLFLOW_URI"])

@testset "`LogMetrics`" begin
    cb = LogMetrics(tbbackend())
    learner = testlearner(Metrics(accuracy), Recorder(), cb)
    @test_nowarn fit!(learner, 1)

    cb = LogMetrics(mlflowbackend())
    learner = testlearner(Metrics(accuracy), Recorder(), cb)
    @test_nowarn fit!(learner, 1)
end


@testset "`LogHyperParams`" begin
    cb = LogHyperParams(tbbackend())
    learner = testlearner(Recorder(), Scheduler(), cb)
    @test_nowarn fit!(learner, 1)
end

@testset "`LogHistograms`" begin
    cb = LogHistograms(tbbackend(), freq = 5)
    learner = testlearner(Recorder(), Scheduler(), cb)
    @test_nowarn fit!(learner, 5)
end

@testset "`LogVisualization`" begin
    cb = LogVisualization(tbbackend(), freq = 5) do batch
        return rand(RGB, 50, 50)
    end
    learner = testlearner(Recorder(), Scheduler(), cb)
    @test_nowarn fit!(learner, 5)
end
