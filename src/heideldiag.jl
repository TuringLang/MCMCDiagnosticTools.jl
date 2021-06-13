"""
    heideldiag(
        x::AbstractVector{<:Real}; alpha::Real=0.05, eps::Real=0.1, start::Int=1, kwargs...
    )

Compute the Heidelberger and Welch diagnostic.
"""
function heideldiag(
    x::AbstractVector{<:Real}; alpha::Real=0.05, eps::Real=0.1, start::Int=1, kwargs...
)
    n = length(x)
    delta = trunc(Int, 0.10 * n)
    y = x[trunc(Int, n / 2):end]
    S0 = length(y) * mcse(y; kwargs...)^2
    i, pvalue, converged, ybar = 1, 1.0, false, NaN
    while i < n / 2
        y = x[i:end]
        m = length(y)
        ybar = Statistics.mean(y)
        B = cumsum(y) - ybar * collect(1:m)
        Bsq = (B .* B) ./ (m * S0)
        I = sum(Bsq) / m
        pvalue = 1.0 - pcramer(I)
        converged = pvalue > alpha
        if converged
            break
        end
        i += delta
    end
    halfwidth = sqrt(2) * SpecialFunctions.erfcinv(alpha) * mcse(y; kwargs...)
    passed = halfwidth / abs(ybar) <= eps
    return (
        burnin=i + start - 2,
        stationarity=converged,
        pvalue=pvalue,
        mean=ybar,
        halfwidth=halfwidth,
        test=passed,
    )
end

## Csorgo S and Faraway JJ. The exact and asymptotic distributions of the
## Cramer-von Mises statistic. Journal of the Royal Statistical Society,
## Series B, 58: 221-234, 1996.
function pcramer(q::Real)
    p = 0.0
    for k in 0:3
        c1 = 4.0 * k + 1.0
        c2 = c1^2 / (16.0 * q)
        p +=
            SpecialFunctions.gamma(k + 0.5) / factorial(k) *
            sqrt(c1) *
            exp(-c2) *
            SpecialFunctions.besselk(0.25, c2)
    end
    return p / (pi^1.5 * sqrt(q))
end
