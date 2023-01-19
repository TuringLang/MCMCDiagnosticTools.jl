using Distributions, Statistics, StatsBase

# AR(1) process
function ar1(φ::Real, σ::Real, n::Int...)
    T = float(Base.promote_eltype(φ, σ))
    x = randn(T, n...)
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
