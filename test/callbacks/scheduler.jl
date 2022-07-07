include("../imports.jl")


@testset "`Scheduler`" begin
    nbatches = 16
    schedule = Interpolator(Poly(λ=1e-4, p=1, max_iter=10), nbatches)
    learner = testlearner(coeff = 3, Recorder(), Scheduler(LearningRate => schedule), nbatches = nbatches)

    epoch!(learner, TrainingPhase())
    @test isapprox(last(learner.cbstate.hyperparams[:LearningRate])[2], 9.0e-5, atol = 0.1)

    epoch!(learner, TrainingPhase())
    @test isapprox(last(learner.cbstate.hyperparams[:LearningRate])[2], 8.0e-5, atol =0.1)

    @testset "Optimisers.jl" begin
        learner = testlearner(
            coeff = 3,
            opt = Optimisers.Descent(0.0001),
            Recorder(), Scheduler(LearningRate => schedule),
            nbatches = nbatches)

        epoch!(learner, TrainingPhase())
        @test isapprox(last(learner.cbstate.hyperparams[:LearningRate])[2], 9.0e-5, atol = 0.1)

        epoch!(learner, TrainingPhase())
        @test isapprox(last(learner.cbstate.hyperparams[:LearningRate])[2], 8.0e-5, atol =0.1)
    end

    @testset "Regression test for #122" begin
        learner = testlearner(Recorder(), Scheduler(LearningRate => ParameterSchedulers.CosAnneal(λ0=0,λ1=0,period=0)), ToGPU())
        @test_nowarn FluxTraining.Graphs.topological_sort_by_dfs(learner.callbacks.graph)
    end
end
