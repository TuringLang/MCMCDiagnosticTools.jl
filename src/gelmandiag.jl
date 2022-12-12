function _gelmandiag(psi::AbstractArray{<:Real,3}; alpha::Real=0.05)
    niters, nchains, nparams = size(psi)
    nchains > 1 || error("Gelman diagnostic requires at least 2 chains")

    rfixed = (niters - 1) / niters
    rrandomscale = (nchains + 1) / (nchains * niters)

    # `eachslice(psi; dims=2)` breaks type inference 
    S2 = map(x -> Statistics.cov(x; dims=1), (view(psi, :, i, :) for i in axes(psi, 2)))
    W = Statistics.mean(S2)

    psibar = dropdims(Statistics.mean(psi; dims=1); dims=1)
    B = niters .* Statistics.cov(psibar)

    w = LinearAlgebra.diag(W)
    b = LinearAlgebra.diag(B)
    s2 = mapreduce(LinearAlgebra.diag, hcat, S2)'
    psibar2 = vec(Statistics.mean(psibar; dims=1))

    var_w = vec(Statistics.var(s2; dims=1)) ./ nchains
    var_b = (2 / (nchains - 1)) .* b .^ 2
    var_wb =
        (niters / nchains) .* (
            LinearAlgebra.diag(Statistics.cov(s2, psibar .^ 2)) .-
            2 .* psibar2 .* LinearAlgebra.diag(Statistics.cov(s2, psibar))
        )

    V = @. rfixed * w + rrandomscale * b
    var_V = rfixed^2 * var_w + rrandomscale^2 * var_b + 2 * rfixed * rrandomscale * var_wb

    df = @. 2 * V^2 / var_V

    B_df = nchains - 1
    W_df = @. 2 * w^2 / var_w

    estimates = Array{Float64}(undef, nparams)
    upperlimits = Array{Float64}(undef, nparams)

    q = 1 - alpha / 2
    for i in 1:nparams
        correction = (df[i] + 3) / (df[i] + 1)
        rrandom = rrandomscale * b[i] / w[i]

        estimates[i] = sqrt(correction * (rfixed + rrandom))

        if !isnan(rrandom)
            rrandom *= Distributions.quantile(Distributions.FDist(B_df, W_df[i]), q)
        end
        upperlimits[i] = sqrt(correction * (rfixed + rrandom))
    end

    return estimates, upperlimits, W, B
end

"""
    gelmandiag(samples::AbstractArray{<:Real,3}; alpha::Real=0.95)

Compute the Gelman, Rubin and Brooks diagnostics [^Gelman1992] [^Brooks1998] on `samples`
with shape `(draws, chains, parameters)`.  Values of the
diagnostic’s potential scale reduction factor (PSRF) that are close to one suggest
convergence.  As a rule-of-thumb, convergence is rejected if the 97.5 percentile of a PSRF
is greater than 1.2.

[^Gelman1992]: Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. Statistical science, 7(4), 457-472.

[^Brooks1998]: Brooks, S. P., & Gelman, A. (1998). General methods for monitoring convergence of iterative simulations. Journal of computational and graphical statistics, 7(4), 434-455.
"""
function gelmandiag(chains::AbstractArray{<:Real,3}; kwargs...)
    estimates, upperlimits = _gelmandiag(chains; kwargs...)

    return (psrf=estimates, psrfci=upperlimits)
end

"""
    gelmandiag_multivariate(samples::AbstractArray{<:Real,3}; alpha::Real=0.05)

Compute the multivariate Gelman, Rubin and Brooks diagnostics on `samples` with shape
`(draws, chains, parameters)`.
"""
function gelmandiag_multivariate(chains::AbstractArray{<:Real,3}; kwargs...)
    niters, nchains, nparams = size(chains)
    if nparams < 2
        error(
            "computation of the multivariate potential scale reduction factor requires ",
            "at least two variables",
        )
    end

    estimates, upperlimits, W, B = _gelmandiag(chains; kwargs...)

    # compute multivariate potential scale reduction factor (PSRF)
    # the eigenvalues of `X := W⁻¹ B` and `Y := L⁻¹ B L⁻ᵀ = L⁻¹ Bᵀ L⁻ᵀ = L⁻¹ (L⁻¹ B)ᵀ`,
    # where `W = L Lᵀ`, are identical but `Y` is symmetric whereas `X` is not in general
    # (remember, `W` and `B` are symmetric positive semi-definite matrices)
    # for symmetric matrices specialized implementations for computing eigenvalues are used
    rfixed = (niters - 1) / niters
    rrandomscale = (nchains + 1) / (nchains * niters)
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(W))
    L = C.L
    Y = L \ (L \ LinearAlgebra.Symmetric(B))'
    λmax = LinearAlgebra.eigmax(LinearAlgebra.Symmetric(Y))
    multivariate = rfixed + rrandomscale * λmax

    return (psrf=estimates, psrfci=upperlimits, psrfmultivariate=multivariate)
end
