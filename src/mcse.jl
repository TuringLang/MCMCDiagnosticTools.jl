"""
    mcse(x::AbstractVector{<:Real}, method::Symbol=:imse; kwargs...)

Return Monte Carlo Standard Errors of samples `x`.
Here, `method` describes how to estimate the errors; possible options are:

- `:bm` for batch means
- `:imse` for the integrated mean-squared (prediction) error
- `:ipse` for the initial positive sequence estimator

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
