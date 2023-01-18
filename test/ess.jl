using Distributions
using DynamicHMC
using LogDensityProblems
using LogExpFunctions
using OffsetArrays
using MCMCDiagnosticTools
using MCMCDiagnosticTools: _rank_normalize
using Random
using Statistics
using StatsBase
using Test

struct ExplicitESSMethod <: MCMCDiagnosticTools.AbstractESSMethod end
struct ExplicitESSCache{S}
    samples::S
end
function MCMCDiagnosticTools.build_cache(::ExplicitESSMethod, samples::Matrix, var::Vector)
    return ExplicitESSCache(samples)
end
MCMCDiagnosticTools.update!(::ExplicitESSCache) = nothing
function MCMCDiagnosticTools.mean_autocov(k::Int, cache::ExplicitESSCache)
    return mean(autocov(cache.samples, k:k; demean=true))
end

struct CauchyProblem end
LogDensityProblems.logdensity(p::CauchyProblem, θ) = -sum(log1psq, θ)
function LogDensityProblems.logdensity_and_gradient(p::CauchyProblem, θ)
    return -sum(log1psq, θ), -2 .* θ ./ (1 .+ θ .^ 2)
end
LogDensityProblems.dimension(p::CauchyProblem) = 50
function LogDensityProblems.capabilities(p::CauchyProblem)
    return LogDensityProblems.LogDensityOrder{1}()
end

@testset "ess.jl" begin
    @testset "ESS and R̂ (IID samples)" begin
        # Repeat tests with different scales
        @testset for scale in (1, 50, 100), nchains in (1, 10), split_chains in (1, 2)
            x = scale * randn(10_000, nchains, 40)
            ntotal = size(x, 1) * size(x, 2)

            ess_standard, rhat_standard = ess_rhat(x; split_chains=split_chains)
            ess_standard2, rhat_standard2 = ess_rhat(
                x; split_chains=split_chains, method=ESSMethod()
            )
            ess_fft, rhat_fft = ess_rhat(
                x; split_chains=split_chains, method=FFTESSMethod()
            )
            ess_bda, rhat_bda = ess_rhat(
                x; split_chains=split_chains, method=BDAESSMethod()
            )

            # check that we get (roughly) the same results
            @test ess_standard == ess_standard2
            @test ess_standard ≈ ess_fft
            @test rhat_standard == rhat_standard2 == rhat_fft == rhat_bda

            # check that the estimates are reasonable
            @test all(x -> isapprox(x, ntotal; rtol=0.1), ess_standard)
            @test all(x -> isapprox(x, ntotal; rtol=0.1), ess_bda)
            @test all(x -> isapprox(x, 1; rtol=0.1), rhat_standard)

            # BDA method fluctuates more
            @test var(ess_standard) < var(ess_bda)
        end
    end

    @testset "ESS and R̂ only promote eltype when necessary" begin
        @testset for T in (Float32, Float64)
            x = rand(T, 100, 4, 2)
            TV = Vector{T}
            @inferred Tuple{TV,TV} ess_rhat(x)
        end
        @testset "Int" begin
            x = rand(1:10, 100, 4, 2)
            TV = Vector{Float64}
            @inferred Tuple{TV,TV} ess_rhat(x)
        end
    end

    @testset "ESS and R̂ are similar vectors to inputs" begin
        # simultaneously checks that we index correctly and that output types are correct
        x = randn(100, 4, 5)
        y = OffsetArray(x, -5:94, 2:5, 11:15)
        S, R = ess_rhat(y)
        @test S isa OffsetVector{Float64}
        @test axes(S, 1) == axes(y, 3)
        @test R isa OffsetVector{Float64}
        @test axes(R, 1) == axes(y, 3)
        S2, R2 = ess_rhat(x)
        @test S2 == collect(S)
        @test R2 == collect(R)
        y = OffsetArray(similar(x, Missing), -5:94, 2:5, 11:15)
        S3, R3 = ess_rhat(y)
        @test S3 isa OffsetVector{Missing}
        @test axes(S3, 1) == axes(y, 3)
        @test R3 isa OffsetVector{Missing}
        @test axes(R3, 1) == axes(y, 3)
    end

    @testset "ESS and R̂ (identical samples)" begin
        x = ones(10_000, 10, 40)

        ess_standard, rhat_standard = ess_rhat(x)
        ess_standard2, rhat_standard2 = ess_rhat(x; method=ESSMethod())
        ess_fft, rhat_fft = ess_rhat(x; method=FFTESSMethod())
        ess_bda, rhat_bda = ess_rhat(x; method=BDAESSMethod())

        # check that the estimates are all NaN
        for ess in (ess_standard, ess_standard2, ess_fft, ess_bda)
            @test all(isnan, ess)
        end
        for rhat in (rhat_standard, rhat_standard2, rhat_fft, rhat_bda)
            @test all(isnan, rhat)
        end
    end

    @testset "ESS and R̂ errors" begin # check that issue #137 is fixed
        x = rand(4, 3, 5)
        x2 = rand(5, 3, 5)
        @test_throws ArgumentError ess_rhat(x; split_chains=1)
        ess_rhat(x2; split_chains=1)
        @test_throws ArgumentError ess_rhat(x2; split_chains=2)
        x3 = rand(100, 3, 5)
        ess_rhat(x3; maxlag=1)
        @test_throws DomainError ess_rhat(x3; maxlag=0)
    end

    @testset "ESS and R̂ with Union{Missing,Float64} eltype" begin
        x = Array{Union{Missing,Float64}}(undef, 1000, 4, 3)
        x .= randn.()
        x[1, 1, 1] = missing
        S, R = ess_rhat(x)
        @test ismissing(S[1])
        @test ismissing(R[1])
        @test !any(ismissing, S[2:3])
        @test !any(ismissing, R[2:3])
    end

    @testset "Autocov of ESSMethod and FFTESSMethod equivalent to StatsBase" begin
        x = randn(1_000, 10, 40)
        ess_exp = ess_rhat(x; method=ExplicitESSMethod())[1]
        @testset "$method" for method in [FFTESSMethod(), ESSMethod()]
            @test ess_rhat(x; method=method)[1] ≈ ess_exp
        end
    end

    @testset "ESS and R̂ for chains with 2 epochs that have not mixed" begin
        # checks that splitting yields lower ESS estimates and higher Rhat estimates
        x = randn(1000, 4, 10) .+ repeat([0, 10]; inner=(500, 1, 1))
        ess_array, rhat_array = ess_rhat(x; split_chains=1)
        @test all(x -> isapprox(x, 1; rtol=0.1), rhat_array)
        ess_array2, rhat_array2 = ess_rhat(x; split_chains=2)
        @test all(ess_array2 .< ess_array)
        @test all(>(2), rhat_array2)
    end

    @testset "ess_rhat(f, x)[1]" begin
        # we check the ESS estimates by simulating uncorrelated, correlated, and
        # anticorrelated chains, mapping the draws to a target distribution, computing the
        # estimand, and estimating the ESS for the chosen estimator, computing the
        # corresponding MCSE, and checking that the mean estimand is close to the asymptotic
        # value of the estimand, with a tolerance chosen using the MCSE.
        ndraws = 100
        nchains = 4
        nparams = 100
        x = randn(ndraws, nchains, nparams)
        mymean(x; kwargs...) = mean(x; kwargs...)
        @test_throws ArgumentError ess_rhat(mymean, x)
        estimators = [mean, median, std, mad, Base.Fix2(quantile, 0.25)]
        dists = [Normal(10, 100), Exponential(10), TDist(7) * 10 - 20]
        # AR(1) coefficients. 0 is IID, -0.3 is slightly anticorrelated, 0.9 is highly autocorrelated
        φs = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        # account for all but the 2 skipped checks
        nchecks = nparams * length(φs) * ((length(estimators) - 1) * length(dists) + 1)
        α = (0.1 / nchecks) / 2  # multiple correction
        @testset for f in estimators, dist in dists, φ in φs
            f === mad && !(dist isa Normal) && continue
            σ = sqrt(1 - φ^2) # ensures stationary distribution is N(0, 1)
            x = ar1(φ, σ, ndraws, nchains, nparams)
            x .= quantile.(dist, cdf.(Normal(), x))  # stationary distribution is dist
            μ_mean = dropdims(mapslices(f ∘ vec, x; dims=(1, 2)); dims=(1, 2))
            dist = asymptotic_dist(f, dist)
            n = @inferred(ess_rhat(f, x))[1]
            μ = mean(dist)
            mcse = sqrt.(var(dist) ./ n)
            for i in eachindex(μ_mean, mcse)
                atol = quantile(Normal(0, mcse[i]), 1 - α)
                @test μ_mean[i] ≈ μ atol = atol
            end
        end
    end

    @testset "ESS thresholded for antithetic chains" begin
        # for φ = -0.3 (slightly antithetic), ESS without thresholding for low ndraws is
        # often >ndraws*log10(ndraws)
        # for φ = -0.9 (highly antithetic), ESS without thresholding for low ndraws is
        # usually negative
        nchains = 4
        @testset for ndraws in (10, 100), φ in (-0.3, -0.9)
            x = ar1(φ, sqrt(1 - φ^2), ndraws, nchains, 1000)
            Smin, Smax = extrema(ess_rhat(mean, x)[1])
            ntotal = ndraws * nchains
            @test Smax == ntotal * log10(ntotal)
            @test Smin > 0
        end
    end

    @testset "ess_rhat_bulk(x)" begin
        xnorm = randn(1_000, 4, 10)
        @test ess_rhat_bulk(xnorm) == ess_rhat(mean, _rank_normalize(xnorm))
        xcauchy = quantile.(Cauchy(), cdf.(Normal(), xnorm))
        # transformation by any monotonic function should not change the bulk ESS/R-hat
        @test ess_rhat_bulk(xnorm) == ess_rhat_bulk(xcauchy)
    end

    @testset "tail- ESS and R-hat detect mismatched scales" begin
        # simulate chains with same stationary mean but different stationary scales
        φ = 0.1 # low autocorrelation
        σs = sqrt(1 - φ^2) .* [0.1, 1, 10, 100]
        ndraws = 1_000
        nparams = 100
        x = 10 .+ mapreduce(hcat, σs) do σ
            return ar1(φ, σ, ndraws, 1, nparams)
        end

        # recommended convergence thresholds
        ess_cutoff = 100 * size(x, 2)  # recommended cutoff is 100 * nchains
        rhat_cutoff = 1.01

        # sanity check that standard and bulk ESS and R-hat both fail to detect
        # mismatched scales
        S, R = ess_rhat(x)
        @test all(≥(ess_cutoff), S)
        @test all(≤(rhat_cutoff), R)
        Sbulk, Rbulk = ess_rhat_bulk(x)
        @test all(≥(ess_cutoff), Sbulk)
        @test all(≤(rhat_cutoff), Rbulk)

        # check that tail-Rhat and tail-ESS detect mismatched scales and signal
        # poor convergence
        S = ess_tail(x)
        @test all(<(ess_cutoff), S)
        R = rhat_tail(x)
        @test all(>(rhat_cutoff), R)
    end

    @testset "bulk and tail ESS and R-hat for heavy tailed" begin
        # sampling Cauchy distribution with large max depth to allow for better tail
        # exploration. From https://avehtari.github.io/rhat_ess/rhat_ess.html chains have
        # okay bulk ESS and R-hat values, while some tail ESS and R-hat values are poor.
        prob = CauchyProblem()
        reporter = NoProgressReport()
        algorithm = DynamicHMC.NUTS(; max_depth=20)
        rng = Random.default_rng()
        posterior_matrices = map(1:4) do _  # ~2.5 mins to sample
            result = mcmc_with_warmup(rng, prob, 1_000; algorithm=algorithm, reporter=reporter)
            hasproperty(result, :posterior_matrix) && return result.posterior_matrix
            return reduce(hcat, result.chain)
        end
        x = permutedims(cat(posterior_matrices...; dims=3), (2, 3, 1))

        Sbulk, Rbulk = ess_rhat_bulk(x)
        Stail = ess_tail(x)
        Rtail = rhat_tail(x)
        ess_cutoff = 100 * size(x, 2)  # recommended cutoff is 100 * nchains
        @test mean(≥(ess_cutoff), Sbulk) > 0.9
        @test mean(≥(ess_cutoff), Stail) < mean(≥(ess_cutoff), Sbulk)
        @test mean(≤(1.01), Rbulk) > 0.9
        @test mean(≤(1.01), Rtail) < 0.8
    end
end
