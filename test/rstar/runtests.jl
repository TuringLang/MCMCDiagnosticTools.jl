using MCMCDiagnosticTools

using Distributions
using MLJBase
using MLJLIBSVMInterface
using MLJXGBoostInterface
using Tables

using Random
using Test

const xgboost_deterministic = Pipeline(XGBoostClassifier(); operation=predict_mode)

@testset "rstar.jl" begin
    classifiers = (XGBoostClassifier(), xgboost_deterministic, SVC())
    N = 1_000

    @testset "samples input type: $wrapper" for wrapper in [Vector, Array, Tables.table]
        @testset "examples (classifier = $classifier)" for classifier in classifiers
            sz = wrapper === Vector ? N : (N, 2)
            # Compute R⋆ statistic for a mixed chain.
            samples = wrapper(randn(sz...))
            dist = rstar(classifier, samples, rand(1:3, N))

            # Mean of the statistic should be focused around 1, i.e., the classifier does not
            # perform better than random guessing.
            if classifier isa MLJBase.Deterministic
                @test dist isa Float64
            else
                @test dist isa LocationScale
                @test dist.ρ isa PoissonBinomial
                @test minimum(dist) == 0
                @test maximum(dist) == 3
            end
            @test mean(dist) ≈ 1 rtol = 0.2
            wrapper === Vector && break

            # Compute R⋆ statistic for a mixed chain.
            samples = wrapper(randn(4 * N, 8))
            chain_indices = repeat(1:4, N)
            dist = rstar(classifier, samples, chain_indices)

            # Mean of the statistic should be closte to 1, i.e., the classifier does not perform
            # better than random guessing.
            if classifier isa MLJBase.Deterministic
                @test dist isa Float64
            else
                @test dist isa LocationScale
                @test dist.ρ isa PoissonBinomial
                @test minimum(dist) == 0
                @test maximum(dist) == 4
            end
            @test mean(dist) ≈ 1 rtol = 0.15

            # Compute the R⋆ statistic for a non-mixed chain.
            samples = wrapper([
                sin.(1:N) cos.(1:N)
                100 .* cos.(1:N) 100 .* sin.(1:N)
            ])
            chain_indices = repeat(1:2; inner=N)
            dist = rstar(classifier, samples, chain_indices)

            # Mean of the statistic should be close to 2, i.e., the classifier should be able to
            # learn an almost perfect decision boundary between chains.
            if classifier isa MLJBase.Deterministic
                @test dist isa Float64
            else
                @test dist isa LocationScale
                @test dist.ρ isa PoissonBinomial
                @test minimum(dist) == 0
                @test maximum(dist) == 2
            end
            @test mean(dist) ≈ 2 rtol = 0.15
        end
        wrapper === Vector && continue

        @testset "exceptions (classifier = $classifier)" for classifier in classifiers
            samples = wrapper(randn(N - 1, 2))
            @test_throws DimensionMismatch rstar(classifier, samples, rand(1:3, N))
            for subset in (-0.3, 0, 1 / (3 * N), 1 - 1 / (3 * N), 1, 1.9)
                samples = wrapper(randn(N, 2))
                @test_throws ArgumentError rstar(
                    classifier, samples, rand(1:3, N); subset=subset
                )
            end
        end
    end

    @testset "table with chain_ids produces same result as 3d array" begin
        nparams = 2
        nchains = 3
        samples = randn(nparams, N, nchains)

        # manually construct samples_mat and chain_inds for comparison
        samples_mat = Matrix{Float64}(undef, N * nchains, nparams)
        chain_inds = Vector{Int}(undef, N * nchains)
        i = 1
        for chain in 1:nchains, draw in 1:N
            samples_mat[i, :] = samples[:, draw, chain]
            chain_inds[i] = chain
            i += 1
        end

        @testset "classifier = $classifier" for classifier in classifiers
            rng = MersenneTwister(42)
            dist1 = rstar(rng, classifier, samples_mat, chain_inds)
            Random.seed!(rng, 42)
            dist2 = rstar(rng, classifier, samples)
            @test dist1 == dist2
            @test typeof(rstar(classifier, samples)) === typeof(dist2)
        end
    end
end
