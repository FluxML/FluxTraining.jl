include("../imports.jl")


@testset ExtendedTestSet "StopEarly" begin
    learner = testlearner(Recorder(), Metrics(), EarlyStopping(1), ProgressPrinter(), coeff = 3, opt = Descent(0.1))
    @test_throws CancelFittingException begin
        @suppress fit!(learner, 100)
    end
    @test_nowarn print(EarlyStopping(1))
    @test_nowarn show(EarlyStopping(FluxTraining.ES.NumberLimit(2)))
end
