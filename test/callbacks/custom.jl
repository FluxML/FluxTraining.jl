include("../imports.jl")


# TODO: fix
#=
@testset ExtendedTestSet "`CustomCallback`" begin
    cb = CustomCallback{EpochEnd,TrainingPhase}((learner) -> CancelFittingException("test"))
    learner = testlearner(coeff = 3, callbacks = [cb])
    fit!(learner, TrainingPhase())
    @test learner.state.history.epochs == 1
end
=#
