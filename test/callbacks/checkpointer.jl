include("../imports.jl")


@testset "`Checkpointer`" begin
    checkpointer = Checkpointer(mktempdir())
    learner = testlearner(Recorder(), Metrics(), checkpointer)
    @test length(readdir(checkpointer.folder)) == 0
    @test_nowarn fit!(learner, 5)
    @test length(readdir(checkpointer.folder)) == 5
end
