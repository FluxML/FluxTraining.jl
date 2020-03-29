using Test
using TestSetExtensions
using Flux
using FluxTraining
using FluxTraining: EpochEnd, LR, getdataloader, getoptimparam

include("./utils.jl")

@testset ExtendedTestSet "Training tests" begin
    @testset ExtendedTestSet "Dummy learner convergence" begin
        learner = dummylearner(3)
        fit!(learner, repeat([TrainingPhase()], 5))
        @test learner.model.factor[1] ≈ 3
        @test learner.phase isa TrainingPhase
    end

    @testset ExtendedTestSet "Recorder" begin
        learner = dummylearner(3)
        r = learner.recorder
        @test r.epoch == 0
        @test r.step == 0
        fit!(learner, [TrainingPhase(), ValidationPhase(), TrainingPhase(), ValidationPhase()])
        @test r.epoch == 2
        @test r.step == length(getdataloader(learner.databunch, TrainingPhase()))
        @test r.steptotal == 2length(getdataloader(learner.databunch, TrainingPhase()))
        @test length(r.stepstats) == 2length(getdataloader(learner.databunch, TrainingPhase()))
        @test length(r.stepstats) == r.steptotal
    end

    @testset ExtendedTestSet "Hyperparameter scheduling" begin
        learner = dummylearner(3)
        setschedule!(learner, Dict(LR => [ParamSchedule(2, 1e-4, 1e-6, anneal_linear)]))
        fit!(learner, TrainingPhase())
        @test getoptimparam(learner.opt, LR) ≈ ((1e-4+1e-6) / 2)
        fit!(learner, TrainingPhase())
        @test getoptimparam(learner.opt, LR) ≈ 1e-6
    end

    @testset ExtendedTestSet "CustomCallback" begin
        cb = CustomCallback{EpochEnd, TrainingPhase}((learner) -> error("test"))
        learner = dummylearner(3, callbacks = [cb])
        @test_throws ErrorException fit!(learner, TrainingPhase())
    end
end
