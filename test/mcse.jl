using Test
using Distributions
using MCMCDiagnosticTools
using OffsetArrays
using Statistics
using StatsBase

@testset "mcse.jl" begin
    @testset "estimator must be provided" begin
        x = randn(100, 4, 10)
        @test_throws MethodError mcse(x)
    end

    @testset "ESS-based methods forward kwargs to ess_rhat" begin
        x = randn(100, 4, 10)
        @testset for f in [mean, median, std, Base.Fix2(quantile, 0.1)]
            @test @inferred(mcse(f, x; split_chains=1)) ≠ mcse(f, x)
        end
    end

    @testset "mcse falls back to mcse_sbm" begin
        x = randn(100, 4, 10)
        @test @inferred(mcse(mad, x)) ==
            mcse_sbm(mad, x) ≠
            mcse_sbm(mad, x; batch_size=16) ==
            mcse(mad, x; batch_size=16)
    end

    @testset "mcse produces similar vectors to inputs" begin
        # simultaneously checks that we index correctly and that output types are correct
        @testset for T in (Float32, Float64),
            f in [mean, median, std, Base.Fix2(quantile, T(0.1)), mad]

            x = randn(T, 100, 4, 5)
            y = OffsetArray(x, -5:94, 2:5, 11:15)
            se = mcse(f, y)
            @test se isa OffsetVector{T}
            @test axes(se, 1) == axes(y, 3)
            se2 = mcse(f, x)
            @test se2 ≈ collect(se)
            # quantile errors if data contains missings
            f isa Base.Fix2{typeof(quantile)} && continue
            y = OffsetArray(similar(x, Missing), -5:94, 2:5, 11:15)
            @test mcse(f, y) isa OffsetVector{Missing}
        end
    end

    @testset "mcse with Union{Missing,Float64} eltype" begin
        x = Array{Union{Missing,Float64}}(undef, 1000, 4, 3)
        x .= randn.()
        x[1, 1, 1] = missing
        @testset for f in [mean, median, std, mad]
            se = mcse(f, x)
            @test ismissing(se[1])
            @test !any(ismissing, se[2:end])
        end
    end

    @testset "estimand is within interval defined by MCSE estimate" begin
        # we check the ESS estimates by simulating uncorrelated, correlated, and
        # anticorrelated chains, mapping the draws to a target distribution, computing the
        # estimand, and estimating the ESS for the chosen estimator, computing the
        # corresponding MCSE, and checking that the mean estimand is close to the asymptotic
        # value of the estimand, with a tolerance chosen using the MCSE.
        ndraws = 100
        nchains = 4
        nparams = 100
        estimators = [mean, median, std, Base.Fix2(quantile, 0.25)]
        dists = [Normal(10, 100), Exponential(10), TDist(7) * 10 - 20]
        mcse_methods = [mcse, mcse_sbm]
        # AR(1) coefficients. 0 is IID, -0.3 is slightly anticorrelated, 0.9 is highly autocorrelated
        φs = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        # account for all but the 2 skipped checks
        nchecks =
            nparams * (length(φs) + count(≤(5), φs)) * length(dists) * length(mcse_methods)
        α = (0.01 / nchecks) / 2  # multiple correction
        @testset for mcse in mcse_methods, f in estimators, dist in dists, φ in φs
            # mcse_sbm underestimates the MCSE for highly correlated chains
            mcse === mcse_sbm && φ > 0.5 && continue
            σ = sqrt(1 - φ^2) # ensures stationary distribution is N(0, 1)
            x = ar1(φ, σ, ndraws, nchains, nparams)
            x .= quantile.(dist, cdf.(Normal(), x))  # stationary distribution is dist
            μ_mean = dropdims(mapslices(f ∘ vec, x; dims=(1, 2)); dims=(1, 2))
            μ = mean(asymptotic_dist(f, dist))
            se = mcse(f, x)
            for i in eachindex(μ_mean, se)
                atol = quantile(Normal(0, se[i]), 1 - α)
                @test μ_mean[i] ≈ μ atol = atol
            end
        end
    end
end
