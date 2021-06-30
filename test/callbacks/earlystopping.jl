include("../imports.jl")


@testset ExtendedTestSet "StopEarly" begin
    learner = testlearner(Recorder(), Metrics(), EarlyStopping(1), ProgressPrinter(), coeff = 3, opt = Descent(0.1))
    @test_throws CancelFittingException fit!(learner, 100)
end
