include("../imports.jl")


@testset ExtendedTestSet "`Recorder`" begin
    learner = testlearner(coeff = 3, Recorder())
    FluxTraining.handle(FluxTraining.Events.Init(), learner, TrainingPhase())
    h = learner.cbstate.history[TrainingPhase()]
    @test h.epochs == 0
    @test h.steps == 0
    fit!(learner, 10)
    h = learner.cbstate.history[TrainingPhase()]
    @test h.epochs == 10
    @test h.stepsepoch == length(getdataiter(TrainingPhase(), learner))
    @test h.steps == 10 * length(getdataiter(TrainingPhase(), learner))
end
