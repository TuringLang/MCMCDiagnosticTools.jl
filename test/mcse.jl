using Test
using Distributions
using MCMCDiagnosticTools
using OffsetArrays
using Statistics
using StatsBase

@testset "mcse.jl" begin
    @testset "estimator defaults to mean" begin
        x = randn(100, 4, 10)
        @test mcse(x) == mcse(x; kind=mean)
    end

    @testset "ESS-based methods forward kwargs to ess" begin
        x = randn(100, 4, 10)
        @testset for f in [mean, median, std, Base.Fix2(quantile, 0.1)]
            @test @inferred(mcse(x; kind=f, split_chains=1)) ≠ mcse(x; kind=f)
        end
    end

    @testset "mcse falls back to _mcse_sbm" begin
        x = randn(100, 4, 10)
        estimator = mad
        @test @inferred(mcse(x; kind=estimator)) ==
            MCMCDiagnosticTools._mcse_sbm(estimator, x) ≠
            MCMCDiagnosticTools._mcse_sbm(estimator, x; batch_size=16) ==
            mcse(x; kind=estimator, batch_size=16)
    end

    @testset "handles all sizes correctly" begin
        @testset for f in [mean, median, std, Base.Fix2(quantile, 0.1), mad]
            x = randn(100, 4, 3, 2)
            se = @inferred mcse(x; kind=f)
            for i in axes(x, 4)
                xi = x[:, :, :, i]
                @test @inferred(mcse(xi; kind=f)) ≈ se[:, i]
                for j in axes(x, 3)
                    xij = xi[:, :, j]
                    @test @inferred(mcse(xij; kind=f)) ≈ se[j, i]
                end
            end
            xv = x[:, 1, 1, 1]
            @test @inferred(mcse(xv; kind=f)) ≈ mcse(reshape(xv, :, 1); kind=f)
        end
    end

    @testset "mcse produces similar vectors to inputs" begin
        # simultaneously checks that we index correctly and that output types are correct
        _axes = (-5:94, 2:5, 11:15, 0:1)
        @testset for T in (Float32, Float64),
            estimator in [mean, median, std, Base.Fix2(quantile, T(0.1)), mad],
            _axes in ((-5:94, 2:5, 11:15), (-5:94, 2:5, 11:15, 0:1))

            N = length(_axes) - 2
            x = randn(T, map(length, _axes)...)
            y = OffsetArray(x, _axes...)
            se = mcse(y; kind=estimator)
            @test se isa OffsetArray{T,N}
            @test axes(se) == _axes[3:end]
            se2 = mcse(x; kind=estimator)
            @test se2 ≈ collect(se)
            # quantile errors if data contains missings
            estimator isa Base.Fix2{typeof(quantile)} && continue
            y = OffsetArray(similar(x, Missing), _axes...)
            @test mcse(y; kind=estimator) isa OffsetArray{Missing,N}
        end
    end

    @testset "mcse with Union{Missing,Float64} eltype" begin
        @testset for sz in ((1000, 4), (1000, 4, 3))
            x = Array{Union{Missing,Float64}}(undef, sz...)
            x .= randn.()
            x[1] = missing
            @testset for f in [mean, median, std, mad]
                se = mcse(x; kind=f)
                if ndims(x) > 2
                    @test ismissing(se[1])
                    @test !any(ismissing, se[2:end])
                else
                    @test ismissing(se)
                end
            end
        end
    end

    @testset "estimand is within interval defined by MCSE estimate" begin
        # we check the MCSE estimates by simulating uncorrelated, correlated, and
        # anticorrelated chains, mapping the draws to a target distribution, computing the
        # estimand, estimating the MCSE for the chosen estimator, and checking that the mean
        # estimand is close to the asymptotic value of the estimand, with a tolerance chosen
        # using the MCSE.
        ndraws = 1000
        nchains = 4
        nparams = 100
        estimators = [mean, median, std, Base.Fix2(quantile, 0.25)]
        dists = [Normal(10, 100), Exponential(10), TDist(7) * 10 - 20]
        mcse_methods = [mcse, MCMCDiagnosticTools._mcse_sbm]
        # AR(1) coefficients. 0 is IID, -0.3 is slightly anticorrelated, 0.9 is highly autocorrelated
        φs = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        # account for all but the 2 skipped checks
        nchecks = nparams * (length(φs) + count(≤(5), φs)) * length(dists)
        α = (0.01 / nchecks) / 2  # multiple correction
        @testset for mcse in mcse_methods, f in estimators, dist in dists, φ in φs
            # _mcse_sbm underestimates the MCSE for highly correlated chains
            mcse === MCMCDiagnosticTools._mcse_sbm && φ > 0.5 && continue
            σ = sqrt(1 - φ^2) # ensures stationary distribution is N(0, 1)
            x = ar1(φ, σ, ndraws, nchains, nparams)
            x .= quantile.(dist, cdf.(Normal(), x))  # stationary distribution is dist
            μ_mean = dropdims(mapslices(f ∘ vec, x; dims=(1, 2)); dims=(1, 2))
            μ = mean(asymptotic_dist(f, dist))
            se = mcse === MCMCDiagnosticTools._mcse_sbm ? mcse(f, x) : mcse(x; kind=f)
            for i in eachindex(μ_mean, se)
                atol = quantile(Normal(0, se[i]), 1 - α)
                @test μ_mean[i] ≈ μ atol = atol
            end
        end
    end
end
