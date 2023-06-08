include("../imports.jl")


@testset "`Checkpointer`" begin
    checkpointer = Checkpointer(mktempdir())
    learner = testlearner(Recorder(), Metrics(), checkpointer)
    @test length(readdir(checkpointer.folder)) == 0
    @test_nowarn fit!(learner, 5)
    @test length(readdir(checkpointer.folder)) == 5
end

@testset "`Checkpointer` with top_k models" begin
    checkpointer = Checkpointer(mktempdir(); keep_top_k=8)
    learner = testlearner(Recorder(), Metrics(), checkpointer)
    @test length(readdir(checkpointer.folder)) == 0
    @test_nowarn fit!(learner, 5)
    @test length(readdir(checkpointer.folder)) == 5
    @test_nowarn fit!(learner, 5)
    @test length(readdir(checkpointer.folder)) == 8

    learner.lossfn = (ŷ, y) -> 1e5
    @test_nowarn fit!(learner, 1)
    # make sure most recent model is still saved, even though it's bad
    most_recent_loss = last(learner.cbstate.metricsepoch[TrainingPhase()], :Loss)[2]
    @assert most_recent_loss == 1e5
    @test any(contains(string(most_recent_loss)),
              Base.Filesystem.readdir(checkpointer.folder))
end
