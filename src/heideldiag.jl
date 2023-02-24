"""
    heideldiag(
        x::AbstractVector{<:Real}; alpha::Real=0.05, eps::Real=0.1, start::Int=1, kwargs...
    )

Compute the Heidelberger and Welch diagnostic [^Heidelberger1983]. This diagnostic tests for
non-convergence (non-stationarity) and whether ratios of estimation interval halfwidths to
means are within a target ratio. Stationarity is rejected (0) for significant test p-values.
Halfwidth tests are rejected (0) if observed ratios are greater than the target, as is the
case for `s2` and `beta[1]`.

`kwargs` are forwarded to [`mcse`](@ref).

[^Heidelberger1983]: Heidelberger, P., & Welch, P. D. (1983). Simulation run length control in the presence of an initial transient. Operations Research, 31(6), 1109-1144.
"""
function heideldiag(
    x::AbstractVector{<:Real}; alpha::Real=1//20, eps::Real=0.1, start::Int=1, kwargs...
)
    n = length(x)
    delta = trunc(Int, 0.10 * n)
    y = x[trunc(Int, n / 2):end]
    T = typeof(zero(eltype(x)) / 1)
    s = first(mcse(reshape(y, :, 1, 1); split_chains=1, kwargs...))
    S0 = length(y) * s^2
    i, pvalue, converged, ybar = 1, one(T), false, T(NaN)
    while i < n / 2
        y = x[i:end]
        m = length(y)
        ybar = Statistics.mean(y)
        B = cumsum(y) - ybar * collect(1:m)
        Bsq = (B .* B) ./ (m * S0)
        I = sum(Bsq) / m
        pvalue = 1 - T(pcramer(I))
        converged = pvalue > alpha
        if converged
            break
        end
        i += delta
    end
    s = first(mcse(reshape(y, :, 1, 1); split_chains=1, kwargs...))
    halfwidth = sqrt2 * SpecialFunctions.erfcinv(T(alpha)) * s
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
