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
