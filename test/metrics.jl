
@testset "Metric function" begin
    @testset "`accuracy`" begin
        @test accuracy([1 0; 0 1], [1 0; 0 1]) == 1
        @test accuracy([0.9 0.1; 0.1 0.9], [1 0; 0 1]) == 1
    end

end
