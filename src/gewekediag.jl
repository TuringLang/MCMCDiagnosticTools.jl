"""
    gewekediag(x::AbstractVector{<:Real}; first::Real=0.1, last::Real=0.5, kwargs...)

Compute the Geweke diagnostic from the `first` and `last` proportion of samples `x`.
"""
function gewekediag(x::AbstractVector{<:Real}; first::Real=0.1, last::Real=0.5, kwargs...)
  0 < first < 1 || throw(ArgumentError("`first` is not in (0, 1)"))
  0 < last < 1 || throw(ArgumentError("`last` is not in (0, 1)"))
  first + last <= 1 || throw(ArgumentError("`first` and `last` proportions overlap"))

  n = length(x)
  x1 = x[1:round(Int, first * n)]
  x2 = x[round(Int, n - last * n + 1):n]
  z = (Statistics.mean(x1) - Statistics.mean(x2)) /
    hypot(mcse(x1; kwargs...), mcse(x2; kwargs...))
  p = SpecialFunctions.erfc(abs(z) / sqrt(2))

  return (zscore=z, pvalue=p)
end
