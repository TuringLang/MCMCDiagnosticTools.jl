#################### Raftery and Lewis Diagnostic ####################

"""
    rafterydiag(x::AbstractVector{<:Real}; q, r, s, eps, range)

Compute the Raftery and Lewis diagnostic.
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
        local test , ntest
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
