using MCMCDiagnosticTools

using Distributions
using EvoTrees
using MLJBase: MLJBase, Pipeline, predict_mode
using MLJDecisionTreeInterface
using MLJLIBSVMInterface
using MLJModels
using MLJXGBoostInterface
using Tables

using Random
using Test

# XGBoost errors on 32bit systems: https://github.com/dmlc/XGBoost.jl/issues/92
const XGBoostClassifiers = if Sys.WORD_SIZE == 64
    (XGBoostClassifier(), Pipeline(XGBoostClassifier(); operation=predict_mode))
else
    ()
end

@testset "rstar.jl" begin
    N = 1_000

    @testset "samples input type: $wrapper" for wrapper in [Vector, Array, Tables.table]
        # In practice, probably you want to use EvoTreeClassifier with early stopping
        classifiers = (
            EvoTreeClassifier(; nrounds=1_000, eta=0.01),
            Pipeline(EvoTreeClassifier(; nrounds=1_000, eta=0.01); operation=predict_mode),
            DecisionTreeClassifier(),
            SVC(),
            XGBoostClassifiers...,
        )
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
                @test maximum(dist) == 6
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
                @test maximum(dist) == 8
            end
            @test mean(dist) ≈ 1 rtol = 0.15

            # Compute the R⋆ statistic for a non-mixed chain.
            samples = wrapper([
                sin.(1:N) cos.(1:N)
                100 .* cos.(1:N) 100 .* sin.(1:N)
            ])
            chain_indices = repeat(1:2; inner=N)
            dist = rstar(classifier, samples, chain_indices; split_chains=1)

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

            # Compute the R⋆ statistic for identical chains that individually have not mixed.
            samples = ones(sz)
            samples[div(N, 2):end, :] .= 2
            chain_indices = repeat(1:4; outer=div(N, 4))
            dist = rstar(classifier, samples, chain_indices; split_chains=1)
            # without split chains cannot distinguish between chains
            @test mean(dist) ≈ 1 rtol = 0.15
            dist = rstar(classifier, samples, chain_indices)
            # with split chains can learn almost perfect decision boundary
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
        nchains = 3
        samples = randn(N, nchains, 2, 4)

        # manually construct samples_mat and chain_inds for comparison
        samples_mat = reshape(samples, N * nchains, size(samples, 3) * size(samples, 4))
        chain_inds = Vector{Int}(undef, N * nchains)
        i = 1
        for chain in 1:nchains, draw in 1:N
            chain_inds[i] = chain
            i += 1
        end

        # In practice, probably you want to use EvoTreeClassifier with early stopping
        rng = MersenneTwister(42)
        classifiers = (
            EvoTreeClassifier(; rng=rng, nrounds=1_000, eta=0.1),
            Pipeline(
                EvoTreeClassifier(; rng=rng, nrounds=1_000, eta=0.1); operation=predict_mode
            ),
            DecisionTreeClassifier(; rng=rng),
            SVC(),
            XGBoostClassifiers...,
        )
        @testset "classifier = $classifier" for classifier in classifiers
            Random.seed!(rng, 42)
            dist1 = rstar(rng, classifier, samples_mat, chain_inds)
            Random.seed!(rng, 42)
            dist2 = rstar(rng, classifier, samples)
            Random.seed!(rng, 42)
            dist3 = rstar(rng, classifier, reshape(samples, N, nchains, :))
            @test dist1 == dist2 == dist3
            @test typeof(rstar(classifier, samples)) === typeof(dist2) === typeof(dist3)
        end
    end

    @testset "model traits requirements" begin
        samples = randn(2, 3, 4)

        inputs_error = ArgumentError(
            "classifier does not support tables of continuous values as inputs"
        )
        model = UnivariateDiscretizer()
        @test_throws inputs_error rstar(model, samples)
        @test_throws inputs_error MCMCDiagnosticTools._check_model_supports_continuous_inputs(
            model
        )

        targets_error = ArgumentError(
            "classifier does not support vectors of multi-class labels as targets"
        )
        predictions_error = ArgumentError(
            "classifier does not support vectors of multi-class labels or their densities as predictions",
        )
        models = if Sys.WORD_SIZE == 64
            (EvoTreeRegressor(), EvoTreeCount(), XGBoostRegressor(), XGBoostCount())
        else
            (EvoTreeRegressor(), EvoTreeCount())
        end
        for model in models
            @test_throws targets_error rstar(model, samples)
            @test_throws targets_error MCMCDiagnosticTools._check_model_supports_multiclass_targets(
                model
            )
            @test_throws predictions_error MCMCDiagnosticTools._check_model_supports_multiclass_predictions(
                model
            )
        end
    end

    @testset "incorrect type of predictions" begin
        @test_throws ArgumentError MCMCDiagnosticTools._rstar(
            AbstractVector{<:MLJBase.Continuous}, rand(2), rand(3)
        )
        @test_throws ArgumentError MCMCDiagnosticTools._rstar(1.0, rand(2), rand(2))
    end

    @testset "single chain: method ambiguity issue" begin
        samples = rand(1:5, N)
        rng = MersenneTwister(42)
        dist = rstar(rng, DecisionTreeClassifier(), samples)
        @test mean(dist) ≈ 1 atol = 0.15
        Random.seed!(rng, 42)
        dist2 = rstar(rng, DecisionTreeClassifier(), samples, ones(Int, N))
        @test dist2 == dist
    end
end
