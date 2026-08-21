const normcdf1 = 0.8413447460685429  # StatsFuns.normcdf(1)
const normcdfn1 = 0.15865525393145705  # StatsFuns.normcdf(-1)

"""
    mcse(samples::AbstractArray{<:Union{Missing,Real}}; kind=Statistics.mean, kwargs...)
    mcse(
        samples::AbstractVector{<:AbstractArray{<:Union{Missing,Real}}};
        kind=Statistics.mean,
        kwargs...,
    )

Estimate the Monte Carlo standard errors (MCSE) of the estimator `kind` applied to `samples`
of shape `(draws, [chains[, parameters...]])`.

$_DOC_RAGGED_SAMPLES

See also: [`ess`](@ref)

## Kinds of MCSE estimates

The estimator whose MCSE should be estimated is specified with `kind`. `kind` must accept a
vector of the same `eltype` as `samples` and return a real estimate.

For the following estimators, the effective sample size [`ess`](@ref) and an estimate
of the asymptotic variance are used to compute the MCSE, and `kwargs` are forwarded to
`ess`:
- [`Statistics.mean`](@extref)
- [`Statistics.median`](@extref)
- [`Statistics.std`](@extref)
- `Base.Fix2(Statistics.quantile, p::Real)`

For other estimators, the subsampling bootstrap method (SBM)[^FlegalJones2011][^Flegal2012]
is used as a fallback, and the only accepted `kwargs` are `batch_size`, which indicates the
size of the overlapping batches used to estimate the MCSE, defaulting to
`floor(Int, sqrt(total_draws))`. Note that SBM tends to underestimate the MCSE,
especially for highly autocorrelated chains. One should verify that autocorrelation is low
by checking the bulk- and tail-ESS values.

[^FlegalJones2011]: Flegal JM, Jones GL. (2011) Implementing MCMC: estimating with confidence.
                    Handbook of Markov Chain Monte Carlo. pp. 175-97.
                    [pdf](http://faculty.ucr.edu/~jflegal/EstimatingWithConfidence.pdf)
[^Flegal2012]: Flegal JM. (2012) Applicability of subsampling bootstrap methods in Markov chain Monte Carlo.
               Monte Carlo and Quasi-Monte Carlo Methods 2010. pp. 363-72.
               doi: [10.1007/978-3-642-27440-4_18](https://doi.org/10.1007/978-3-642-27440-4_18)

"""
function mcse(x::_DiagnosticSamples; kind=Statistics.mean, kwargs...)
    _validate_samples(x)
    return _mcse(kind, x; kwargs...)
end

_mcse(f, x; kwargs...) = _mcse_sbm(f, x; kwargs...)
function _mcse(::typeof(Statistics.mean), samples::_DiagnosticSamples; kwargs...)
    S = _ess(Statistics.mean, samples; kwargs...)
    return _sample_std(samples) ./ sqrt.(S)
end
function _mcse(::typeof(Statistics.std), samples::_DiagnosticSamples; kwargs...)
    x = _squared_deviations(samples)  # expectand proxy
    S = _ess(Statistics.mean, x; kwargs...)
    # asymptotic variance of sample variance estimate is Var[var] = E[μ₄] - E[var]²,
    # where μ₄ is the 4th central moment
    # by the delta method, Var[std] = Var[var] / 4E[var] = (E[μ₄]/E[var] - E[var])/4,
    # See e.g. Chapter 3 of Van der Vaart, AW. (200) Asymptotic statistics. Vol. 3.
    mean_var = _sample_mean(x)
    mean_moment4 = _sample_mean(abs2, x)
    return @. sqrt((mean_moment4 / mean_var - mean_var) / S) / 2
end
function _mcse(
    f::Base.Fix2{typeof(Statistics.quantile),<:Real}, samples::_DiagnosticSamples; kwargs...
)
    p = f.x
    S = _ess(f, samples; kwargs...)
    return _mcse_from_quantile_ess(samples, p, S)
end
function _mcse(::typeof(Statistics.median), samples::_DiagnosticSamples; kwargs...)
    S = _ess(Statistics.median, samples; kwargs...)
    return _mcse_from_quantile_ess(samples, 1//2, S)
end

function _mcse_from_quantile_ess(samples::_DiagnosticSamples, p, S)
    isempty(_param_axes(samples)) && return _mcse_quantile(_pool(samples), p, S)
    T = eltype(S)
    E = _sample_eltype(samples)
    R = promote_type(E, typeof(oneunit(E) / sqrt(oneunit(T))))
    values = similar(S, R)
    for (i, xi) in zip(eachindex(values, S), _eachparam(samples))
        values[i] = _mcse_quantile(_pool(xi), p, S[i])
    end
    return values
end

function _mcse_quantile(x, p, Seff)
    Seff === missing && return missing
    if isnan(Seff)
        return oftype(oneunit(eltype(x)) / 1, NaN)
    end
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
    f, x::_DiagnosticSamples; batch_size::Int=floor(Int, sqrt(_total_draws(x)))
)
    isempty(_param_axes(x)) && return _mcse_sbm(f, _pool(x), batch_size)
    T0 = _sample_eltype(x)
    T = promote_type(T0, typeof(zero(T0) / 1))
    values = _similar_params(x, T)
    for (i, xi) in zip(eachindex(values), _eachparam(x))
        values[i] = _mcse_sbm(f, _pool(xi), batch_size)
    end
    return values
end
function _mcse_sbm(f, x, batch_size)
    any(x -> x === missing, x) && return missing
    n = length(x)
    i1 = firstindex(x)
    if allequal(x)
        y1 = f(view(x, i1:(i1 + batch_size - 1)))
        return oftype(y1, NaN)
    end
    v = Statistics.var(
        f(view(x, i:(i + batch_size - 1))) for i in i1:(i1 + n - batch_size);
        corrected=false,
    )
    return sqrt(v * (batch_size//n))
end
