using Test
using MCMCDiagnosticTools
using Statistics
using StatsBase

@testset "mcse.jl" begin
    @testset "estimand is within interval defined by MCSE estimate" begin
        # we check the ESS estimates by simulating uncorrelated, correlated, and
        # anticorrelated chains, mapping the draws to a target distribution, computing the
        # estimand, and estimating the ESS for the chosen estimator, computing the
        # corresponding MCSE, and checking that the mean estimand is close to the asymptotic
        # value of the estimand, with a tolerance chosen using the MCSE.
        ndraws = 1_000
        nchains = 4
        nparams = 100
        estimators = [mean, median, std, Base.Fix2(quantile, 0.25)]
        dists = [Normal(10, 100), Exponential(10), TDist(7) * 10 - 20]
        mcse_methods = [mcse, mcse_sbm]
        # AR(1) coefficients. 0 is IID, -0.3 is slightly anticorrelated, 0.9 is highly autocorrelated
        φs = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        # account for all but the 2 skipped checks
        nchecks =
            nparams * length(φs) * length(estimators) * length(dists) * length(mcse_methods)
        α = (0.1 / nchecks) / 2  # multiple correction
        @testset for mcse in mcse_methods, f in estimators, dist in dists, φ in φs
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
