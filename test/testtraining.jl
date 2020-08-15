using Test
using TestSetExtensions
using Flux
using FluxTraining
using FluxTraining: EpochEnd, LR, getdataloader, getoptimparam
using MLDataPattern

include("./imports.jl")


@testset ExtendedTestSet "Dummy learner convergence" begin
    learner = dummylearner(3)
    fit!(learner, repeat([TrainingPhase()], 5))
    @test learner.model.factor[1] ≈ 3
    @test learner.state.phase isa TrainingPhase
end

@testset ExtendedTestSet "Recorder" begin
    learner = dummylearner(3)
    h = learner.state.history
    @test h.epochs == 0
    @test h.nsteps == 0
    fit!(learner, [TrainingPhase(), ValidationPhase(), TrainingPhase(), ValidationPhase()])
    @test h.epochs == 2
    @test h.nstepsepoch == length(getdataloader(TrainingPhase(), learner))
    @test h.nsteps == 2 * length(getdataloader(TrainingPhase(), learner))

    loss1, loss2 = get(h.epochmetrics[ValidationPhase()], :loss)[2]
    @test loss1 > loss2
end

@testset ExtendedTestSet "Hyperparameter scheduling" begin

    schedules = Schedules(Dict(LR => Schedule(16, 1e-4, 1e-6, anneal_linear)))
    learner = dummylearner(3, schedule = schedules)

    fit!(learner, TrainingPhase())
    @test getoptimparam(learner.opt, LR) ≈ ((1e-4 + 1e-6) / 2)

    fit!(learner, TrainingPhase())
    @test getoptimparam(learner.opt, LR) ≈ 1e-6
end
