const normcdf1 = 0.8413447460685429  # StatsFuns.normcdf(1)
const normcdfn1 = 0.15865525393145705  # StatsFuns.normcdf(-1)

"""
    mcse(samples::AbstractArray{<:Union{Missing,Real}}; estimator=Statistics.mean, kwargs...)

Estimate the Monte Carlo standard errors (MCSE) of the `estimator` applied to `samples` of
shape `(draws, chains, parameters)`.

See also: [`ess_rhat`](@ref)

## Estimators

`estimator` must accept a vector of the same `eltype` as `samples` and return a real estimate.

For the following estimators, the effective sample size [`ess_rhat`](@ref) and an estimate
of the asymptotic variance are used to compute the MCSE, and `kwargs` are forwarded to
`ess_rhat`:
- `Statistics.mean`
- `Statistics.median`
- `Statistics.std`
- `Base.Fix2(Statistics.quantile, p::Real)`

For other estimators, the subsampling bootstrap method (SBM)[^FlegalJones2011][^Flegal2012]
is used as a fallback, and the only accepted `kwargs` are `batch_size`, which indicates the
size of the overlapping batches used to estimate the MCSE, defaulting to
`floor(Int, sqrt(draws * chains))`. Note that SBM tends to underestimate the MCSE,
especially for highly autocorrelated chains. One should verify that autocorrelation is low
by checking the bulk- and tail-[`ess_rhat`](@ref) values.

[^FlegalJones2011]: Flegal JM, Jones GL. (2011) Implementing MCMC: estimating with confidence.
                    Handbook of Markov Chain Monte Carlo. pp. 175-97.
                    [pdf](http://faculty.ucr.edu/~jflegal/EstimatingWithConfidence.pdf)
[^Flegal2012]: Flegal JM. (2012) Applicability of subsampling bootstrap methods in Markov chain Monte Carlo.
               Monte Carlo and Quasi-Monte Carlo Methods 2010. pp. 363-72.
               doi: [10.1007/978-3-642-27440-4_18](https://doi.org/10.1007/978-3-642-27440-4_18)

"""
function mcse(
    x::AbstractArray{<:Union{Missing,Real},3}; estimator=Statistics.mean, kwargs...
)
    return _mcse(estimator, x; kwargs...)
end

_mcse(f, x; kwargs...) = _mcse_sbm(f, x; kwargs...)
function _mcse(
    ::typeof(Statistics.mean), samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...
)
    S = ess(samples; estimator=Statistics.mean, kwargs...)
    return dropdims(Statistics.std(samples; dims=(1, 2)); dims=(1, 2)) ./ sqrt.(S)
end
function _mcse(
    ::typeof(Statistics.std), samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...
)
    x = (samples .- Statistics.mean(samples; dims=(1, 2))) .^ 2  # expectand proxy
    S = ess(x; estimator=Statistics.mean, kwargs...)
    # asymptotic variance of sample variance estimate is Var[var] = E[μ₄] - E[var]²,
    # where μ₄ is the 4th central moment
    # by the delta method, Var[std] = Var[var] / 4E[var] = (E[μ₄]/E[var] - E[var])/4,
    # See e.g. Chapter 3 of Van der Vaart, AW. (200) Asymptotic statistics. Vol. 3.
    mean_var = dropdims(Statistics.mean(x; dims=(1, 2)); dims=(1, 2))
    mean_moment4 = dropdims(Statistics.mean(abs2, x; dims=(1, 2)); dims=(1, 2))
    return @. sqrt((mean_moment4 / mean_var - mean_var) / S) / 2
end
function _mcse(
    f::Base.Fix2{typeof(Statistics.quantile),<:Real},
    samples::AbstractArray{<:Union{Missing,Real},3};
    kwargs...,
)
    p = f.x
    S = ess(samples; estimator=f, kwargs...)
    T = eltype(S)
    R = promote_type(eltype(samples), typeof(oneunit(eltype(samples)) / sqrt(oneunit(T))))
    values = similar(S, R)
    for (i, xi, Si) in zip(eachindex(values), eachslice(samples; dims=3), S)
        values[i] = _mcse_quantile(vec(xi), p, Si)
    end
    return values
end
function _mcse(
    ::typeof(Statistics.median), samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...
)
    S = ess(samples; estimator=Statistics.median, kwargs...)
    T = eltype(S)
    R = promote_type(eltype(samples), typeof(oneunit(eltype(samples)) / sqrt(oneunit(T))))
    values = similar(S, R)
    for (i, xi, Si) in zip(eachindex(values), eachslice(samples; dims=3), S)
        values[i] = _mcse_quantile(vec(xi), 1//2, Si)
    end
    return values
end

function _mcse_quantile(x, p, Seff)
    Seff === missing && return missing
    S = length(x)
    # quantile error distribution is asymptotically normal; estimate σ (mcse) with 2
    # quadrature points: xl and xu, chosen as quantiles so that xu - xl = 2σ
    # compute quantiles of error distribution in probability space (i.e. quantiles passed through CDF)
    # Beta(α,β) is the approximate error distribution of quantile estimates
    α = Seff * p + 1
    β = Seff * (1 - p) + 1
    prob_x_upper = StatsFuns.betainvcdf(α, β, normcdf1)
    prob_x_lower = StatsFuns.betainvcdf(α, β, normcdfn1)
    # use inverse ECDF to get quantiles in quantile (x) space
    l = max(floor(Int, prob_x_lower * S), 1)
    u = min(ceil(Int, prob_x_upper * S), S)
    iperm = partialsortperm(x, l:u)  # sort as little of x as possible
    xl = x[first(iperm)]
    xu = x[last(iperm)]
    # estimate mcse from quantiles
    return (xu - xl) / 2
end

function _mcse_sbm(
    f,
    x::AbstractArray{<:Union{Missing,Real},3};
    batch_size::Int=floor(Int, sqrt(size(x, 1) * size(x, 2))),
)
    T = promote_type(eltype(x), typeof(zero(eltype(x)) / 1))
    values = similar(x, T, (axes(x, 3),))
    for (i, xi) in zip(eachindex(values), eachslice(x; dims=3))
        values[i] = _mcse_sbm(f, vec(xi), batch_size)
    end
    return values
end
function _mcse_sbm(f, x, batch_size)
    any(x -> x === missing, x) && return missing
    n = length(x)
    i1 = firstindex(x)
    v = Statistics.var(
        f(view(x, i:(i + batch_size - 1))) for i in i1:(i1 + n - batch_size);
        corrected=false,
    )
    return sqrt(v * (batch_size//n))
end
