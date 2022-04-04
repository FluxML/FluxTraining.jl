
include("../imports.jl")

@testset "`GarbageCollect`" begin
    cb = GarbageCollect()
    learner = testlearner(Recorder(), Metrics(), cb)
    @test_nowarn fit!(learner, 1)
end
