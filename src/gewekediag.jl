"""
    gewekediag(x::AbstractVector{<:Real}; first::Real=0.1, last::Real=0.5, kwargs...)

Compute the Geweke diagnostic [^Geweke1991] from the `first` and `last` proportion of
samples `x`.

The diagnostic is designed to asses convergence of posterior means estimated with
autocorrelated samples.  It computes a normal-based test statistic comparing the sample
means in two windows containing proportions of the first and last iterations.  Users should
ensure that there is sufficient separation between the two windows to assume that their
samples are independent.  A non-significant test p-value indicates convergence.  Significant
p-values indicate non-convergence and the possible need to discard initial samples as a
burn-in sequence or to simulate additional samples.

[^Geweke1991]: Geweke, J. F. (1991). Evaluating the accuracy of sampling-based approaches to the calculation of posterior moments (No. 148). Federal Reserve Bank of Minneapolis.
"""
function gewekediag(x::AbstractVector{<:Real}; first::Real=0.1, last::Real=0.5, kwargs...)
    0 < first < 1 || throw(ArgumentError("`first` is not in (0, 1)"))
    0 < last < 1 || throw(ArgumentError("`last` is not in (0, 1)"))
    first + last <= 1 || throw(ArgumentError("`first` and `last` proportions overlap"))

    n = length(x)
    x1 = x[1:round(Int, first * n)]
    x2 = x[round(Int, n - last * n + 1):n]
    z =
        (Statistics.mean(x1) - Statistics.mean(x2)) /
        hypot(mcse(x1; kwargs...), mcse(x2; kwargs...))
    p = SpecialFunctions.erfc(abs(z) / sqrt(2))

    return (zscore=z, pvalue=p)
end
