include("./imports.jl")



@testset ExtendedTestSet "Dummy learner convergence" begin
    learner = testlearner(coeff = 3)
    fit!(learner, repeat([TrainingPhase()], 5))
    @test  learner.model.coeff[1] ≈ 3 atol = 0.1
    @test learner.state.phase isa TrainingPhase
end


@testset ExtendedTestSet "Recorder" begin
    learner = testlearner(coeff = 3)
    h = learner.state.history
    @test h.epochs == 0
    @test h.nsteps == 0
    fit!(learner, 10)
    @test h.epochs == 10
    @test h.nstepsepoch == length(getdataloader(TrainingPhase(), learner))
    @test h.nsteps == 10 * length(getdataloader(TrainingPhase(), learner))

    idxs, losses = get(h.epochmetrics[ValidationPhase()], :loss)

    @test losses[1] > losses[end]
end


@testset ExtendedTestSet "Hyperparameter scheduling" begin
    nbatches = 16
    schedules = Schedules(Dict(LR => Schedule(2nbatches, 1e-4, 1e-6, anneal_linear)))
    learner = testlearner(coeff = 3, nbatches = nbatches, schedule = schedules)

    fit!(learner, TrainingPhase())
    @test getoptimparam(learner.opt, LR) ≈ ((1e-4 + 1e-6) / 2)

    fit!(learner, TrainingPhase())
    @test getoptimparam(learner.opt, LR) ≈ 1e-6
end
