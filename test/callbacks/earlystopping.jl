include("../imports.jl")


@testset "StopEarly" begin
    learner = testlearner(Recorder(), Metrics(), EarlyStopping(1), ProgressPrinter(), coeff = 3, opt = Descent(0.1))
    @test_throws CancelFittingException begin
        @suppress fit!(learner, 100)
    end
    @test_nowarn print(Base.DevNull(), EarlyStopping(1))
    @test_nowarn show(Base.DevNull(), EarlyStopping(FluxTraining.ES.NumberLimit(2)))
end
