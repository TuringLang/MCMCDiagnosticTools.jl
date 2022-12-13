using MCMCDiagnosticTools
using Test
using Random
using Statistics

@testset "split_chains" begin
    x = rand(100, 4, 8)
    @test @inferred(MCMCDiagnosticTools.split_chains(x, 1)) == x
    @test MCMCDiagnosticTools.split_chains(x, 2) == reshape(x, 50, 8, 8)
    @test MCMCDiagnosticTools.split_chains(x, 3) == reshape(x[1:99, :, :], 33, 12, 8)
end

@testset "rank_normalize" begin
    x = randn(1000) ./ randn.()  # cauchy draws
    z = @inferred MCMCDiagnosticTools.rank_normalize(x)
    @test size(z) == size(x)
    @test mean(z) ≈ 0 atol=1e-13
    @test std(z) ≈ 1 rtol=1e-2

    x = randexp(1000, 4, 8)
    @test_broken @inferred MCMCDiagnosticTools.rank_normalize(x)
    z = MCMCDiagnosticTools.rank_normalize(x)
    @test size(z) == size(x)
    @test all(xi -> isapprox(xi, 0; atol=1e-13), mean(z; dims=(1, 2)))
    @test all(xi -> isapprox(xi, 1; rtol=1e-2), std(z; dims=(1, 2)))
end
