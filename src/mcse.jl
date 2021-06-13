#################### Monte Carlo Standard Errors ####################

"""
    mcse(x::AbstractVector{<:Real}; method::Symbol=:imse, kwargs...)

Compute the Monte Carlo standard error (MCSE) of samples `x`.
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
    mbar = [Statistics.mean(@view(x[(i + 1):(i + m)])) for i in 0:m:(n - m)]
    return StatsBase.sem(mbar)
end

function mcse_imse(x::AbstractVector{<:Real})
    n = length(x)
    m = div(n - 2, 2)
    x_ = map(Float64, x)
    ghat = StatsBase.autocov(x_, [0, 1])
    Ghat = sum(ghat)
    value = -ghat[1] + 2 * Ghat
    for i in 1:m
        Ghat = min(Ghat, sum(StatsBase.autocov(x_, [2 * i, 2 * i + 1])))
        Ghat > 0 || break
        value += 2 * Ghat
    end
    return sqrt(value / n)
end

function mcse_ipse(x::AbstractVector{<:Real})
    n = length(x)
    m = div(n - 2, 2)
    x_ = map(Float64, x)
    ghat = StatsBase.autocov(x_, [0, 1])
    value = ghat[1] + 2 * ghat[2]
    for i in 1:m
        Ghat = sum(StatsBase.autocov(x_, [2 * i, 2 * i + 1]))
        Ghat > 0 || break
        value += 2 * Ghat
    end
    return sqrt(value / n)
end
