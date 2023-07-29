using Test
using MCMCDiagnosticTools

@testset "p_direction.jl" begin
    @testset "Correct result" begin
        @test p_direction([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]) == 0.6
    end
end