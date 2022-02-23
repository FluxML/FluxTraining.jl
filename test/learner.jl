include("imports.jl")


@testset "Learner callback utilities" begin
    @testset "setcallbacks!" begin
        learner = Learner(nothing, nothing, nothing, nothing; usedefaultcallbacks = false)
        @test length(learner.callbacks.cbs) == 0
        FluxTraining.setcallbacks!(learner, [Metrics()])
        @test length(learner.callbacks.cbs) == 1
    end

    @testset "addcallback!" begin
        learner = Learner(nothing, nothing, nothing, nothing; usedefaultcallbacks = false)
        @test length(learner.callbacks.cbs) == 0
        FluxTraining.addcallback!(learner, Metrics())
        @test length(learner.callbacks.cbs) == 1
    end

    @testset "getcallback" begin
        learner = Learner(nothing, nothing, nothing, nothing; usedefaultcallbacks = false)
        @test isnothing(FluxTraining.getcallback(learner, Metrics))
        FluxTraining.addcallback!(learner, Metrics())
        @test !isnothing(FluxTraining.getcallback(learner, Metrics))
    end

    @testset "replacecallback!" begin
        learner = Learner(nothing, nothing, nothing, nothing, Metrics(accuracy); usedefaultcallbacks = false)
        @test length(learner.callbacks.cbs[1].metrics) == 2
        FluxTraining.replacecallback!(learner, Metrics())
        @test length(learner.callbacks.cbs[1].metrics) == 1
    end
end
