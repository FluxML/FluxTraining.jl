include("../imports.jl")


@testset ExtendedTestSet "Metric" begin
    cb = Metrics(Metric(accuracy, phase = ValidationPhase))
    learner = testlearner(Recorder(), cb)
    @test_nowarn fit!(learner, 1)
    @test :Accuracy ∈ keys(learner.cbstate.metricsstep[ValidationPhase()])
    @test !(:Accuracy ∈ keys(learner.cbstate.metricsstep[TrainingPhase()]))
    @test :Accuracy ∈ keys(learner.cbstate.metricsepoch[ValidationPhase()])
    @test !(:Accuracy ∈ keys(learner.cbstate.metricsepoch[TrainingPhase()]))
end
