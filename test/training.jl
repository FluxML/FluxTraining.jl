include("./imports.jl")



@testset ExtendedTestSet "runstep, runepoch" begin
    learner = testlearner(coeff = 3)
    struct TestPhase <: Phase end
    state = runstep(learner, TestPhase()) do handle, state
        state.xs, state.ys = learner.data.training[1]
        state.loss = sum(abs.(state.xs - state.ys))
    end
    @test state.xs isa Vector
    @test state.loss isa Number
    @test_nowarn runepoch(learner, TestPhase()) do handle
        throw(FluxTraining.CancelEpochException("test"))
    end
end


@testset ExtendedTestSet "TrainingPhase" begin
    learner = testlearner(coeff = 3)
    runepoch(learner, TrainingPhase()) do _
        state = step!(learner, TrainingPhase(), learner.data.training[1])
        @test state.xs isa Vector
        @test state.ys isa Vector
        @test state.ŷs isa Vector
        @test state.grads isa Zygote.Grads
        @test state.loss isa Number
    end
end


@testset ExtendedTestSet "ValidationPhase" begin
    learner = testlearner(coeff = 3)
    runepoch(learner, ValidationPhase()) do _
        state = step!(learner, ValidationPhase(), learner.data.validation[1])
        @test state.xs isa Vector
        @test state.ys isa Vector
        @test state.ŷs isa Vector
        @test state.loss isa Number
    end
end


@testset ExtendedTestSet "Basic `Learner` convergence" begin
    learner = testlearner(coeff = 3)
    fit!(learner, 5)
    @test learner.model.coeff[1] ≈ 3 atol = 0.1
end
