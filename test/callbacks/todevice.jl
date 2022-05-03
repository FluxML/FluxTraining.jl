@testset "ToDevice" begin
    cb = ToDevice(cpu, cpu)
    learner = testlearner(coeff = 3., cb)
    @test_nowarn fit!(learner, 3)
end
