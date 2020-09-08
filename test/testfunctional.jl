

@testset ExtendedTestSet "Annealing" begin
    @testset ExtendedTestSet "Edge identity" begin
        for anneal_fn in (anneal_const, anneal_cosine, anneal_exp, anneal_linear)
            @test anneal_fn(0, 0.1, 1) ≈ 0.1
            @test anneal_fn(1, 0.1, 1) ≈ 1
        end
    end
    @testset ExtendedTestSet "Exponential annealing undefined from 0" begin
        @test_throws DomainError anneal_exp(0, 0, 1)
    end
end


@testset ExtendedTestSet "Functional metrics" begin
    @test accuracy([1 0; 0 1], [1 0; 0 1]) == 1
    @test accuracy([0.9 0.1; 0.1 0.9], [1 0; 0 1]) == 1

end
