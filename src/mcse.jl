"""
    mcse(samples::AbstractVector{<:Real}; method::Symbol=:imse, kwargs...)
    mcse(samples::AbstractArray{<:Real,3}; method::Symbol=:imse, kwargs...)

Compute the Monte Carlo standard error (MCSE) of `samples` of shape `(draws,)` or
`(parameters, draws, chains)`
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
function mcse(x::AbstractArray{<:Real,3}; kwargs...)
    return dropdims(mapslices(xi -> mcse(vec(xi); kwargs...), x; dims=(2, 3)); dims=(2, 3))
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
