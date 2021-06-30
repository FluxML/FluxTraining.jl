include("../imports.jl")


@testset ExtendedTestSet "`Recorder`" begin
    learner = testlearner(coeff = 3, Recorder())
    h = learner.cbstate.history[TrainingPhase()]
    @test h.epochs == 0
    @test h.steps == 0
    fit!(learner, 10)
    h = learner.cbstate.history[TrainingPhase()]
    @test h.epochs == 10
    @test h.stepsepoch == length(learner.data.training)
    @test h.steps == 10 * length(learner.data.training)
end
