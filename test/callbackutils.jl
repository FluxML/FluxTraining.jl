include("imports.jl")

@testset "getcallback" begin
    learner = testlearner(usedefaultcallbacks = true)
    @test getcallback(learner, Metrics) isa Metrics
    @test getcallback(learner, Checkpointer) isa Nothing
end

@testset "removecallback!" begin
    learner = testlearner(usedefaultcallbacks = true)
    @test removecallback!(learner, Metrics) isa Metrics
    @test getcallback(learner, Metrics) isa Nothing

end

@testset "replacecallback!" begin
    learner = testlearner(usedefaultcallbacks = true)
    cb = getcallback(learner, Metrics)
    cbtaken = replacecallback!(learner, Metrics(accuracy))
    @test cb === cbtaken
    removecallback!(learner, Metrics)
    @test replacecallback!(learner, cb) isa Nothing

end


@testset "addcallback!" begin
    learner = testlearner(usedefaultcallbacks = true)
    cb = Checkpointer(mktempdir())
    @test getcallback(learner, Checkpointer) isa Nothing
    @test_nowarn addcallback!(learner, cb)
    @test getcallback(learner, Checkpointer) isa Checkpointer
end


@testset "setcallbacks!" begin
    learner = testlearner(usedefaultcallbacks = true)
    @test length(learner.callbacks.cbs) != 1
    @test_nowarn setcallbacks!(learner, [Metrics(accuracy)])
    @test length(learner.callbacks.cbs) == 1
end
