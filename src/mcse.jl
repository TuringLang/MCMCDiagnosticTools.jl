Base.@irrational normcdf1 0.8413447460685429486 StatsFuns.normcdf(big(1))
Base.@irrational normcdfn1 0.1586552539314570514 StatsFuns.normcdf(big(-1))

"""
    mcse(x::AbstractVector{<:Real}; method::Symbol=:imse, kwargs...)

Compute the Monte Carlo standard error (MCSE) of samples `x`.
The optional argument `method` describes how the errors are estimated. Possible options are:

- `:bm` for batch means [^Glynn1991]
- `:imse` initial monotone sequence estimator [^Geyer1992]
- `:ipse` initial positive sequence estimator [^Geyer1992]

[^Glynn1991]: Glynn, P. W., & Whitt, W. (1991). Estimating the asymptotic variance with batch means. Operations Research Letters, 10(8), 431-435.

[^Geyer1992]: Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. Statistical Science, 473-483.
"""
function mcse(x::AbstractVector{<:Real}; method::Symbol=:imse, kwargs...)
    return if method === :bm
        mcse_bm(x; kwargs...)
    elseif method === :imse
        mcse_imse(x)
    elseif method === :ipse
        mcse_ipse(x)
    else
        throw(ArgumentError("unsupported MCSE method $method"))
    end
end

function mcse_bm(x::AbstractVector{<:Real}; size::Int=floor(Int, sqrt(length(x))))
    n = length(x)
    m = min(div(n, 2), size)
    m == size || @warn "batch size was reduced to $m"
    mcse = StatsBase.sem(Statistics.mean(@view(x[(i + 1):(i + m)])) for i in 0:m:(n - m))
    return mcse
end

function mcse_imse(x::AbstractVector{<:Real})
    n = length(x)
    lags = [0, 1]
    ghat = StatsBase.autocov(x, lags)
    Ghat = sum(ghat)
    @inbounds value = Ghat + ghat[2]
    @inbounds for i in 2:2:(n - 2)
        lags[1] = i
        lags[2] = i + 1
        StatsBase.autocov!(ghat, x, lags)
        Ghat = min(Ghat, sum(ghat))
        Ghat > 0 || break
        value += 2 * Ghat
    end

    mcse = sqrt(value / n)

    return mcse
end

function mcse_ipse(x::AbstractVector{<:Real})
    n = length(x)
    lags = [0, 1]
    ghat = StatsBase.autocov(x, lags)
    @inbounds value = ghat[1] + 2 * ghat[2]
    @inbounds for i in 2:2:(n - 2)
        lags[1] = i
        lags[2] = i + 1
        StatsBase.autocov!(ghat, x, lags)
        Ghat = sum(ghat)
        Ghat > 0 || break
        value += 2 * Ghat
    end

    mcse = sqrt(value / n)

    return mcse
end

function mcse(
    ::typeof(Statistics.mean), samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...
)
    S = ess_rhat(Statistics.mean, samples; kwargs...)[1]
    return dropdims(Statistics.std(samples; dims=(1, 2)); dims=(1, 2)) ./ sqrt.(S)
end
function mcse(
    ::typeof(Statistics.std), samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...
)
    x = (samples .- Statistics.mean(samples; dims=(1, 2))) .^ 2
    S = ess_rhat(Statistics.mean, x; kwargs...)[1]
    mean_var = dropdims(Statistics.mean(x; dims=(1, 2)); dims=(1, 2))
    mean_moment4 = dropdims(Statistics.mean(abs2, x; dims=(1, 2)); dims=(1, 2))
    return @. sqrt((mean_moment4 / mean_var - mean_var) / S) / 2
end
function mcse(
    f::Base.Fix2{typeof(Statistics.quantile),<:Real},
    samples::AbstractArray{<:Union{Missing,Real},3};
    kwargs...,
)
    p = f.x
    S = ess_rhat(f, samples; kwargs...)[1]
    T = eltype(S)
    R = promote_type(eltype(samples), typeof(oneunit(eltype(samples)) / sqrt(oneunit(T))))
    values = similar(S, R)
    map!(values, eachslice(samples; dims=3), S) do xi, Si
        return _mcse_quantile(vec(xi), p, Si)
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
function mcse(
    ::typeof(Statistics.median), samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...
)
    return mcse(Base.Fix2(Statistics.quantile, 1//2), samples; kwargs...)
end

"""
    mcse_sbm(estimator, samples::AbstractArray{<:Union{Missing,Real},3}; batch_size)

Estimate the Monte Carlo standard errors (MCSE) of the `estimator` appplied to `samples`
using the subsampling bootstrap method.

`samples` has shape `(draws, chains, parameters)`, and `estimator` must accept a vector of
the same eltype as `x` and return a real estimate.

`batch_size` indicates the size of the overlapping batches used to estimate the MCSE,
defaulting to `floor(Int, sqrt(draws * chains))`.
"""
function mcse_sbm(
    f,
    x::AbstractArray{<:Union{Missing,Real},3};
    batch_size::Int=floor(Int, sqrt(size(x, 1) * size(x, 2))),
)
    T = promote_type(eltype(x), typeof(zero(eltype(x)) / 1))
    values = similar(x, T, (axes(x, 3),))
    map!(values, eachslice(x; dims=3)) do xi
        return _mcse_sbm(f, vec(xi); batch_size=batch_size)
    end
    return values
end
function _mcse_sbm(f, x; batch_size)
    n = length(x)
    i1 = firstindex(x)
    v = Statistics.var(
        f(view(x, i:(i + size - 1))) for i in i1:(i1 + n - batch_size); corrected=false
    )
    return sqrt(v * (batch_size//n))
end
