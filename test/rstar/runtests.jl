using InferenceDiagnostics
using MLJModels

using Statistics
using Test

@testset "rstar.jl" begin
    XGBoost = @load XGBoostClassifier
    classif = XGBoost()
    N = 1_000

    # Compute R* statistic for a mixed chain.
    samples = randn(N, 2)
    R = rstar(classif, randn(N, 2), rand(1:3, N))

    # Resulting R value should be close to one, i.e. the classifier does not perform better than random guessing.
    @test Statistics.mean(R) ≈ 1 rtol=0.15

    # Compute R* statistic for a mixed chain.
    samples = randn(4 * N, 8)
    chain_indices = repeat(1:4, N)
    R = rstar(classif, samples, chain_indices)

    # Resulting R value should be close to one, i.e. the classifier does not perform better than random guessing.
    @test Statistics.mean(R) ≈ 1 rtol=0.15

    # Compute R* statistic for a non-mixed chain.
    samples = [sin.(1:N) cos.(1:N);
               100 .* cos.(1:N) 100 .* sin.(1:N)]
    chain_indices = repeat(1:2; inner=N)

    # Restuling R value should be close to two, i.e. the classifier should be able to learn an almost perfect decision boundary between chains.
    R = rstar(classif, samples, chain_indices)
    @test Statistics.mean(R) ≈ 2 rtol=5e-2
end
