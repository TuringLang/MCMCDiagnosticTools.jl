using Distributions
using MCMCDiagnosticTools
using Random
using Statistics
using StatsBase
using Test

# AR(1) process
function ar1(rng::AbstractRNG, φ::Real, σ::Real, n::Int...)
    T = float(Base.promote_eltype(φ, σ))
    x = randn(rng, T, n...)
    x .*= σ
    accumulate!(x, x; dims=1) do xi, ϵ
        return muladd(φ, xi, ϵ)
    end
    return x
end

asymptotic_dist(::typeof(mean), dist) = Normal(mean(dist), std(dist))
function asymptotic_dist(::typeof(var), dist)
    μ = var(dist)
    σ = μ * sqrt(kurtosis(dist) + 2)
    return Normal(μ, σ)
end
function asymptotic_dist(::typeof(std), dist)
    μ = std(dist)
    σ = μ * sqrt(kurtosis(dist) + 2) / 2
    return Normal(μ, σ)
end
asymptotic_dist(::typeof(median), dist) = asymptotic_dist(Base.Fix2(quantile, 1//2), dist)
function asymptotic_dist(f::Base.Fix2{typeof(quantile),<:Real}, dist)
    p = f.x
    μ = quantile(dist, p)
    σ = sqrt(p * (1 - p)) / pdf(dist, μ)
    return Normal(μ, σ)
end
function asymptotic_dist(::typeof(mad), dist::Normal)
    # Example 21.10 of Asymptotic Statistics. Van der Vaart
    d = Normal(zero(dist.μ), dist.σ)
    dtrunc = truncated(d; lower=0)
    μ = median(dtrunc)
    σ = 1 / (4 * pdf(d, quantile(d, 3//4)))
    return Normal(μ, σ) / quantile(Normal(), 3//4)
end

@testset "ess.jl" begin
    @testset "ESS and R̂ (IID samples)" begin
        rawx = randn(10_000, 10, 40)

        # Repeat tests with different scales
        for scale in (1, 50, 100)
            x = scale * rawx

            ess_standard, rhat_standard = ess_rhat(x)
            ess_standard2, rhat_standard2 = ess_rhat(x; method=ESSMethod())
            ess_fft, rhat_fft = ess_rhat(x; method=FFTESSMethod())
            ess_bda, rhat_bda = ess_rhat(x; method=BDAESSMethod())

            # check that we get (roughly) the same results
            @test ess_standard == ess_standard2
            @test ess_standard ≈ ess_fft
            @test rhat_standard == rhat_standard2 == rhat_fft == rhat_bda

            # check that the estimates are reasonable
            @test all(x -> isapprox(x, 100_000; rtol=0.1), ess_standard)
            @test all(x -> isapprox(x, 100_000; rtol=0.1), ess_bda)
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

    @testset "ESS and R̂ (single sample)" begin # check that issue #137 is fixed
        x = rand(1, 3, 5)

        for method in (ESSMethod(), FFTESSMethod(), BDAESSMethod())
            # analyze array
            ess_array, rhat_array = ess_rhat(x; method=method)

            @test length(ess_array) == size(x, 3)
            @test all(ismissing, ess_array) # since min(maxlag, niter - 1) = 0
            @test length(rhat_array) == size(x, 3)
            @test all(ismissing, rhat_array)
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
        ndraws = 1_000
        nchains = 4
        nparams = 100
        estimators = [mean, median, std, mad, Base.Fix2(quantile, 0.25)]
        dists = [Normal(10, 100), Exponential(10), TDist(7) * 10 - 20]
        # AR(1) coefficients. 0 is IID, -0.3 is slightly anticorrelated, 0.9 is highly autocorrelated
        φs = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
        iter = filter(collect(Iterators.product(estimators, dists, φs))) do (f, dist, φ)
            return !(f === mad) || dist isa Normal
        end
        nchecks = nparams * length(iter)
        α = (0.1 / nchecks) / 2  # multiple correction
        rng = Random.default_rng()
        @testset "f=$f, dist=$dist, φ=$φ" for (f, dist, φ) in iter
            f === mad && !(dist isa Normal) && continue
            σ = sqrt(1 - φ^2) # ensures stationary distribution is N(0, 1)
            x = ar1(rng, φ, σ, ndraws, nchains, nparams)
            x .= quantile.(dist, cdf.(Normal(), x))  # stationary distribution is dist
            μ_mean = dropdims(mapslices(f ∘ vec, x; dims=(1, 2)); dims=(1, 2))
            dist = asymptotic_dist(f, dist)
            n = ess_rhat(f, x)[1]
            μ = mean(dist)
            mcse = sqrt.(var(dist) ./ n)
            for i in eachindex(μ_mean, mcse)
                atol = quantile(Normal(0, mcse[i]), 1 - α)
                @test μ_mean[i] ≈ μ atol=atol
            end
        end
    end
end
