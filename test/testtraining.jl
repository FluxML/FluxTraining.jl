using Test
using TestSetExtensions
using Flux
using Training
using Training: getdataloader


@testset ExtendedTestSet "Training tests" begin
    @testset ExtendedTestSet "Dummy learner convergence" begin
        learner = dummylearner(3)
        fit!(learner, repeat([TrainingPhase()], 5))
        @test learner.model.factor[1] ≈ 3
        @test learner.phase isa TrainingPhase
    end

    @testset ExtendedTestSet "Recorder" begin
        learner = dummylearner(3)
        @test learner.recorder.epoch == 0
        @test learner.recorder.step == 0
        fit!(learner, [TrainingPhase(), TrainingPhase()])
        @test learner.recorder.epoch == 2
        @test learner.recorder.step == length(getdataloader(learner.databunch, TrainingPhase()))
        @test learner.recorder.steptotal == 2*length(getdataloader(learner.databunch, TrainingPhase()))
    end

    @testset ExtendedTestSet "Hyperparameter scheduling" begin
        learner = dummylearner(3)
        setschedule!(learner, Dict(Training.LR => [ParamSchedule(1, 1e-4, 1e-6, anneal_linear)]))
        fit!(learner, TrainingPhase())
        @test Training.getoptimparam(learner.opt, Training.LR) ≈ 1e-6
    end

    @testset ExtendedTestSet "CustomCallback" begin
        cb = CustomCallback{Training.EpochEnd, TrainingPhase}((learner) -> error("test"))
        learner = dummylearner(3, callbacks = [cb])
        @test_throws ErrorException fit!(learner, TrainingPhase())
    end
end
