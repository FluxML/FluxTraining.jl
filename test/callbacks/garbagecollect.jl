
include("../imports.jl")

@testset ExtendedTestSet "`Checkpointer`" begin
    cb = GarbageCollect()
    learner = testlearner(Recorder(), Metrics(), cb)
    @test_nowarn fit!(learner, 1)
end
