"""
    rhat(samples::AbstractArray{Union{Real,Missing},3}; type=:rank, split_chains=2)

Compute the ``\\widehat{R}`` diagnostics for each parameter in `samples` of shape
`(chains, draws, parameters)`. [^VehtariGelman2021]

`type` indicates the type of ``\\widehat{R}`` to compute (see below).

$_DOC_SPLIT_CHAINS

See also [`ess`](@ref), [`ess_rhat`](@ref), [`rstar`](@ref)

## Types

The following types are supported:
- `:rank`: maximum of ``\\widehat{R}`` with `type=:bulk` and `type=:tail`.
- `:bulk`: basic ``\\widehat{R}``` computed on rank-normalized draws. This type diagnoses
    poor convergence in the bulk of the distribution due to trends or different locations of
    the chains.
- `:tail`: ``\\widehat{R}`` computed on draws folded around the median and then
    rank-normalized. This type diagnoses poor convergence in the tails of the distribution
    due to different scales of the chains.
- `:basic`: Classic ``\\widehat{R}``.

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
"""
@constprop :aggressive function rhat(
    samples::AbstractArray{<:Union{Missing,Real},3}; type=Val(:rank), kwargs...
)
    return _rhat(_val(type), samples; kwargs...)
end

function _rhat(
    ::Val{:basic}, chains::AbstractArray{<:Union{Missing,Real},3}; split_chains::Int=2
)
    # compute size of matrices (each chain may be split!)
    niter = size(chains, 1) ÷ split_chains
    nchains = split_chains * size(chains, 2)
    axes_out = (axes(chains, 3),)
    T = promote_type(eltype(chains), typeof(zero(eltype(chains)) / 1))

    # define output arrays
    rhat = similar(chains, T, axes_out)

    T === Missing && return rhat

    # define caches for mean and variance
    chain_mean = Array{T}(undef, 1, nchains)
    chain_var = Array{T}(undef, nchains)
    samples = Array{T}(undef, niter, nchains)

    # compute correction factor
    correctionfactor = (niter - 1)//niter

    # for each parameter
    for (i, chains_slice) in zip(eachindex(rhat), eachslice(chains; dims=3))
        # check that no values are missing
        if any(x -> x === missing, chains_slice)
            rhat[i] = missing
            continue
        end

        # split chains
        copyto_split!(samples, chains_slice)

        # calculate mean of chains
        Statistics.mean!(chain_mean, samples)

        # calculate within-chain variance
        @inbounds for j in 1:nchains
            chain_var[j] = Statistics.var(
                view(samples, :, j); mean=chain_mean[j], corrected=true
            )
        end
        W = Statistics.mean(chain_var)

        # compute variance estimator var₊, which accounts for between-chain variance as well
        # avoid NaN when nchains=1 and set the variance estimator var₊ to the the within-chain variance in that case
        var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected=(nchains > 1))

        # estimate rhat
        rhat[i] = sqrt(var₊ / W)
    end

    return rhat
end
_rhat(::Val{:bulk}, x; kwargs...) = _rhat(Val(:basic), _rank_normalize(x); kwargs...)
_rhat(::Val{:tail}, x; kwargs...) = _rhat(Val(:bulk), _fold_around_median(x); kwargs...)
function _rhat(::Val{:rank}, x; kwargs...)
    Rbulk = _rhat(Val(:bulk), x; kwargs...)
    Rtail = _rhat(Val(:tail), x; kwargs...)
    return map(max, Rtail, Rbulk)
end
