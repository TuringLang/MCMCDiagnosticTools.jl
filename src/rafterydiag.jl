@doc raw"""
    rafterydiag(
        x::AbstractVector{<:Real}; q=0.025, r=0.005, s=0.95, eps=0.001, range=1:length(x)
    )

Compute the Raftery and Lewis diagnostic [^Raftery1992]. This diagnostic is used to
determine the number of iterations required to estimate a specified quantile `q` within a
desired degree of accuracy.  The diagnostic is designed to determine the number of
autocorrelated samples required to estimate a specified quantile $\theta_q$, such that
$\Pr(\theta \le \theta_q) = q$, within a desired degree of accuracy. In particular, if
$\hat{\theta}_q$ is the estimand and $\Pr(\theta \le \hat{\theta}_q) = \hat{P}_q$ the
estimated cumulative probability, then accuracy is specified in terms of `r` and `s`, where
$\Pr(q - r < \hat{P}_q < q + r) = s$. Thinning may be employed in the calculation of the
diagnostic to satisfy its underlying assumptions. However, users may not want to apply the
same (or any) thinning when estimating posterior summary statistics because doing so results
in a loss of information. Accordingly, sample sizes estimated by the diagnostic tend to be
conservative (too large).

Furthermore, the argument `r` specifies the margin of error for estimated cumulative
probabilities and `s` the probability for the margin of error. `eps` specifies the tolerance
within which the probabilities of transitioning from initial to retained iterations are
within the equilibrium probabilities for the chain. This argument determines the number of
samples to discard as a burn-in sequence and is typically left at its default value.

[^Raftery1992]: A L Raftery and S Lewis. Bayesian Statistics, chapter How Many Iterations in the Gibbs Sampler? Volume 4. Oxford University Press, New York, 1992.
"""
function rafterydiag(
    x::AbstractVector{<:Real}; q=0.025, r=0.005, s=0.95, eps=0.001, range=1:length(x)
)
    nx = length(x)
    phi = sqrt(2.0) * SpecialFunctions.erfinv(s)
    nmin = ceil(Int, q * (1.0 - q) * (phi / r)^2)
    if nmin > nx
        @warn "At least $nmin samples are needed for specified q, r, and s"
        kthin = -1
        burnin = total = NaN
    else
        dichot = Int[(x .<= StatsBase.quantile(x, q))...]
        kthin = 0
        bic = 1.0
        local test, ntest
        while bic >= 0.0
            kthin += 1
            test = dichot[1:kthin:nx]
            ntest = length(test)
            temp = test[1:(ntest - 2)] + 2 * test[2:(ntest - 1)] + 4 * test[3:ntest]
            trantest = reshape(StatsBase.counts(temp, 0:7), 2, 2, 2)
            g2 = 0.0
            for i1 in 1:2, i2 in 1:2, i3 in 1:2
                tt = trantest[i1, i2, i3]
                if tt > 0
                    fitted =
                        sum(trantest[:, i2, i3]) * sum(trantest[i1, i2, :]) /
                        sum(trantest[:, i2, :])
                    g2 += 2.0 * tt * log(tt / fitted)
                end
            end
            bic = g2 - 2.0 * log(ntest - 2.0)
        end

        tranfinal = StatsBase.counts(test[1:(ntest - 1)] + 2 * test[2:ntest], 0:3)
        alpha = tranfinal[3] / (tranfinal[1] + tranfinal[3])
        beta = tranfinal[2] / (tranfinal[2] + tranfinal[4])
        kthin *= step(range)
        m = log(eps * (alpha + beta) / max(alpha, beta)) / log(abs(1.0 - alpha - beta))
        burnin = kthin * ceil(m) + first(range) - 1
        n = ((2.0 - alpha - beta) * alpha * beta * phi^2) / (r^2 * (alpha + beta)^3)
        keep = kthin * ceil(n)
        total = burnin + keep
    end
    return (
        thinning=kthin, burnin=burnin, total=total, nmin=nmin, dependencefactor=total / nmin
    )
end
