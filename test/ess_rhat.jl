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

mymean(x) = mean(x)

@testset "ess_rhat.jl" begin
    @testset "ess/ess_rhat/rhat basics" begin
        @testset "only promote eltype when necessary" begin
            @testset for kind in (:rank, :bulk, :tail, :basic)
                @testset for T in (Float32, Float64)
                    x = rand(T, 100, 4, 2)
                    TV = Vector{T}
                    kind === :rank || @test @inferred(ess(x; kind=kind)) isa TV
                    @test @inferred(rhat(x; kind=kind)) isa TV
                    @test @inferred(ess_rhat(x; kind=kind)) isa Tuple{TV,TV}
                end
                @testset "Int" begin
                    x = rand(1:10, 100, 4, 2)
                    TV = Vector{Float64}
                    kind === :rank || @test @inferred(ess(x; kind=kind)) isa TV
                    @test @inferred(rhat(x; kind=kind)) isa TV
                    @test @inferred(ess_rhat(x; kind=kind)) isa Tuple{TV,TV}
                end
            end
            @testset for kind in [mean, median, mad, std, Base.Fix2(quantile, 0.25)]
                @testset for T in (Float32, Float64)
                    x = rand(T, 100, 4, 2)
                    @test @inferred(ess(x; kind=kind)) isa Vector{T}
                end
                @testset "Int" begin
                    x = rand(1:10, 100, 4, 2)
                    @test @inferred(ess(x; kind=kind)) isa Vector{Float64}
                end
            end
        end

        @testset "errors" begin # check that issue #137 is fixed
            x = rand(4, 3, 5)
            x2 = rand(5, 3, 5)
            x3 = rand(100, 3, 5)
            @testset for f in [ess, ess_rhat]
                @testset for kind in [:rank, :bulk, :tail, :basic]
                    f === ess && kind === :rank && continue
                    @test_throws ArgumentError f(x; split_chains=1, kind=kind)
                    f(x2; split_chains=1, kind=kind)
                    @test_throws ArgumentError f(x2; split_chains=2, kind=kind)
                    f(x3; maxlag=1, kind=kind)
                    @test_throws DomainError f(x3; maxlag=0, kind=kind)
                end
                @test_throws ArgumentError f(x2; kind=:foo)
            end
            @test_throws ArgumentError rhat(x2; kind=:foo)
            @test_throws ArgumentError ess(x2; kind=mymean)
        end

        @testset "Union{Missing,Float64} eltype" begin
            @testset for kind in [:rank, :bulk, :tail, :basic]
                x = Array{Union{Missing,Float64}}(undef, 1000, 4, 3)
                x .= randn.()
                x[1, 1, 1] = missing
                S1 = ess(x; kind=kind === :rank ? :bulk : kind)
                R1 = rhat(x; kind=kind)
                S2, R2 = ess_rhat(x; kind=kind)
                @test ismissing(S1[1])
                @test ismissing(R1[1])
                @test ismissing(S2[1])
                @test ismissing(R2[1])
                @test !any(ismissing, S1[2:3])
                @test !any(ismissing, R1[2:3])
                @test !any(ismissing, S2[2:3])
                @test !any(ismissing, R2[2:3])
            end
        end

        @testset "produces similar vectors to inputs" begin
            @testset for kind in [:rank, :bulk, :tail, :basic]
                # simultaneously checks that we index correctly and that output types are correct
                x = randn(100, 4, 5)
                y = OffsetArray(x, -5:94, 2:5, 11:15)
                S11 = ess(y; kind=kind === :rank ? :bulk : kind)
                R11 = rhat(y; kind=kind)
                S12, R12 = ess_rhat(y; kind=kind)
                @test S11 isa OffsetVector{Float64}
                @test S12 isa OffsetVector{Float64}
                @test axes(S11, 1) == axes(S12, 1) == axes(y, 3)
                @test R11 isa OffsetVector{Float64}
                @test R12 isa OffsetVector{Float64}
                @test axes(R11, 1) == axes(R12, 1) == axes(y, 3)
                S21 = ess(x; kind=kind === :rank ? :bulk : kind)
                R21 = rhat(x; kind=kind)
                S22, R22 = ess_rhat(x; kind=kind)
                @test S22 == S21 == collect(S21)
                @test R21 == R22 == collect(R11)
                y = OffsetArray(similar(x, Missing), -5:94, 2:5, 11:15)
                S31 = ess(y; kind=kind === :rank ? :bulk : kind)
                R31 = rhat(y; kind=kind)
                S32, R32 = ess_rhat(y; kind=kind)
                @test S31 isa OffsetVector{Missing}
                @test S32 isa OffsetVector{Missing}
                @test axes(S31, 1) == axes(S32, 1) == axes(y, 3)
                @test R31 isa OffsetVector{Missing}
                @test R32 isa OffsetVector{Missing}
                @test axes(R31, 1) == axes(R32, 1) == axes(y, 3)
            end
        end

        @testset "ess, ess_rhat, and rhat consistency" begin
            x = randn(1000, 4, 10)
            @testset for kind in [:rank, :bulk, :tail, :basic], split_chains in [1, 2]
                R1 = rhat(x; kind=kind, split_chains=split_chains)
                @testset for method in [ESSMethod(), BDAESSMethod()], maxlag in [100, 10]
                    S1 = ess(
                        x;
                        kind=kind === :rank ? :bulk : kind,
                        split_chains=split_chains,
                        method=method,
                        maxlag=maxlag,
                    )
                    S2, R2 = ess_rhat(
                        x;
                        kind=kind,
                        split_chains=split_chains,
                        method=method,
                        maxlag=maxlag,
                    )
                    @test S1 == S2
                    @test R1 == R2
                end
            end
        end
    end

    # now that we have checked mutual consistency of each method, we perform all following
    # checks for whichever method is most convenient

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

    @testset "Autocov of ESSMethod and FFTESSMethod equivalent to StatsBase" begin
        x = randn(1_000, 10, 40)
        ess_exp = ess(x; method=ExplicitESSMethod())
        @testset "$method" for method in [FFTESSMethod(), ESSMethod()]
            @test ess(x; method=method) ≈ ess_exp
        end
    end

    @testset "ESS and R̂ for chains with 2 epochs that have not mixed" begin
        # checks that splitting yields lower ESS estimates and higher Rhat estimates
        x = randn(1000, 4, 10) .+ repeat([0, 10]; inner=(500, 1, 1))
        ess_array, rhat_array = ess_rhat(x; kind=:basic, split_chains=1)
        @test all(x -> isapprox(x, 1; rtol=0.1), rhat_array)
        ess_array2, rhat_array2 = ess_rhat(x; kind=:basic, split_chains=2)
        @test all(ess_array2 .< ess_array)
        @test all(>(2), rhat_array2)
    end

    @testset "ess(x; kind=f)" begin
        # we check the ESS estimates by simulating uncorrelated, correlated, and
        # anticorrelated chains, mapping the draws to a target distribution, computing the
        # estimand, and estimating the ESS for the chosen estimator, computing the
        # corresponding MCSE, and checking that the mean estimand is close to the asymptotic
        # value of the estimand, with a tolerance chosen using the MCSE.
        ndraws = 1000
        nchains = 4
        nparams = 100
        x = randn(ndraws, nchains, nparams)
        mymean(x; kwargs...) = mean(x; kwargs...)
        @test_throws ArgumentError ess(x; kind=mymean)
        estimators = [mean, median, std, mad, Base.Fix2(quantile, 0.25)]
        dists = [Normal(10, 100), Exponential(10), TDist(7) * 10 - 20]
        # AR(1) coefficients. 0 is IID, -0.3 is slightly anticorrelated, 0.9 is highly autocorrelated
        φs = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        # account for all but the 2 skipped checks
        nchecks = nparams * length(φs) * ((length(estimators) - 1) * length(dists) + 1)
        α = (0.01 / nchecks) / 2  # multiple correction
        @testset for f in estimators, dist in dists, φ in φs
            f === mad && !(dist isa Normal) && continue
            σ = sqrt(1 - φ^2) # ensures stationary distribution is N(0, 1)
            x = ar1(φ, σ, ndraws, nchains, nparams)
            x .= quantile.(dist, cdf.(Normal(), x))  # stationary distribution is dist
            μ_mean = dropdims(mapslices(f ∘ vec, x; dims=(1, 2)); dims=(1, 2))
            dist = asymptotic_dist(f, dist)
            n = @inferred(ess(x; kind=f))
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
            Smin, Smax = extrema(ess(x; kind=mean))
            ntotal = ndraws * nchains
            @test Smax == ntotal * log10(ntotal)
            @test Smin > 0
        end
    end

    @testset "ess(x; kind=:bulk)" begin
        xnorm = randn(1_000, 4, 10)
        @test ess(xnorm; kind=:bulk) == ess(_rank_normalize(xnorm); kind=:basic)
        xcauchy = quantile.(Cauchy(), cdf.(Normal(), xnorm))
        # transformation by any monotonic function should not change the bulk ESS/R-hat
        @test ess(xnorm; kind=:bulk) == ess(xcauchy; kind=:bulk)
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
        S, R = ess_rhat(x; kind=:basic)
        @test all(≥(ess_cutoff), S)
        @test all(≤(rhat_cutoff), R)
        Sbulk, Rbulk = ess_rhat(x; kind=:bulk)
        @test all(≥(ess_cutoff), Sbulk)
        @test all(≤(rhat_cutoff), Rbulk)

        # check that tail- ESS detects mismatched scales and signal poor convergence
        S_tail, R_tail = ess_rhat(x; kind=:tail)
        @test all(<(ess_cutoff), S_tail)
        @test all(>(rhat_cutoff), R_tail)
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

        Sbulk, Rbulk = ess_rhat(x; kind=:bulk)
        Stail, Rtail = ess_rhat(x; kind=:tail)
        ess_cutoff = 100 * size(x, 2)  # recommended cutoff is 100 * nchains
        @test mean(≥(ess_cutoff), Sbulk) > 0.9
        @test mean(≥(ess_cutoff), Stail) < mean(≥(ess_cutoff), Sbulk)
        @test mean(≤(1.01), Rbulk) > 0.9
        @test mean(≤(1.01), Rtail) < 0.8
    end
end
