using InferenceDiagnostics

using Distributions
using MLJModels

using Test

@testset "rstar.jl" begin
    XGBoost = @load XGBoostClassifier verbosity = 0
    SVC = @load SVC verbosity = 0
    classifiers = (XGBoost(), SVC())
    N = 1_000

    @testset "examples (classifier = $classifier)" for classifier in classifiers
        # Compute distribution of R⋆ statistic for a mixed chain.
        samples = randn(N, 2)
        dist = rstar(classifier, randn(N, 2), rand(1:3, N))

        # Mean of the resulting distribution should be focused around 1, i.e., the
        # classifier does not perform better than random guessing.
        if classifier isa MLJModels.Deterministic
            @test dist isa Dirac
            @test 0 <= minimum(dist) <= maximum(dist) <= 3
        else
            @test dist isa LocationScale
            @test dist.ρ isa PoissonBinomial
            @test minimum(dist) == 0
            @test maximum(dist) == 3
        end
        @test mean(dist) ≈ 1 rtol = 0.15

        # Compute distribution of R⋆ statistic for a mixed chain.
        samples = randn(4 * N, 8)
        chain_indices = repeat(1:4, N)
        dist = rstar(classifier, samples, chain_indices)

        # Mean of the resulting distribution should be closte to 1, i.e., the classifier
        # does not perform better than random guessing.
        if classifier isa MLJModels.Deterministic
            @test dist isa Dirac
            @test 0 <= minimum(dist) <= maximum(dist) <= 4
        else
            @test dist isa LocationScale
            @test dist.ρ isa PoissonBinomial
            @test minimum(dist) == 0
            @test maximum(dist) == 4
        end
        @test mean(dist) ≈ 1 rtol = 0.15

        # Compute distribution of R⋆ statistic for a non-mixed chain.
        samples = [
            sin.(1:N) cos.(1:N)
            100 .* cos.(1:N) 100 .* sin.(1:N)
        ]
        chain_indices = repeat(1:2; inner=N)
        dist = rstar(classifier, samples, chain_indices)

        # Mean of the resulting distribution should be close to 2, i.e., the classifier
        # should be able to learn an almost perfect decision boundary between chains.
        if classifier isa MLJModels.Deterministic
            @test dist isa Dirac
            @test 0 <= minimum(dist) == maximum(dist) <= 2
        else
            @test dist isa LocationScale
            @test dist.ρ isa PoissonBinomial
            @test minimum(dist) == 0
            @test maximum(dist) == 2
        end
        @test mean(dist) ≈ 2 rtol = 0.15
    end

    @testset "exceptions (classifier = $classifier)" for classifier in classifiers
        @test_throws DimensionMismatch rstar(classifier, randn(N - 1, 2), rand(1:3, N))
        for subset in (-0.3, 0, 1 / (3 * N), 1 - 1 / (3 * N), 1, 1.9)
            @test_throws ArgumentError rstar(
                classifier, randn(N, 2), rand(1:3, N); subset=subset
            )
        end
    end
end
