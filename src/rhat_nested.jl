"""
    rhat_nested(
        samples::AbstractArray{<:Union{Missing,Real}},
        superchain_ids::AbstractVector;
        split_chains=2,
        kind=:rank,
    )

Compute the nested ``\\widehat{R}`` diagnostic for each parameter in `samples` of shape
`(draws, chains[, parameters...])`.[^Margossian2024]

Nested ``\\widehat{R}`` is a useful convergence diagnostic when running many short chains.
It is calculated on superchains, which are groups of chains that have been initialized at
the same point.

`superchain_ids` is a vector of length `chains` specifying to which superchain each chain
belongs. Each superchain must have the same number of chains. All chains within the same
superchain are assumed to have been initialized at the same point, and there must be at
least 2 superchains.

`kind` indicates the kind of ``\\widehat{R}`` to compute (see extended help).

See also [`rhat`](@ref), [`ess_rhat`](@ref), [`ess`](@ref)

[^Margossian2024]: Margossian, C. C., Hoffman, M. D., Sountsov, P., Riou-Durand, L.,
    Vehtari, A., & Gelman, A. (2024). Nested ``\\widehat{R}``: Assessing the convergence
    of Markov chain Monte Carlo when running many short chains. Bayesian Analysis.
    doi: [10.1214/24-BA1453](https://doi.org/10.1214/24-BA1453)
    arXiv: [2110.13017](https://arxiv.org/abs/2110.13017)

# Extended Help

$_DOC_SPLIT_CHAINS

$_DOC_RHAT_KIND

!!! note
    There is a slight difference in the calculation of ``\\widehat{R}`` and nested
    ``\\widehat{R}``, as nested ``\\widehat{R}`` is lower bounded by 1. This means that
    nested ``\\widehat{R}`` with one chain per superchain will not be exactly equal to
    the usual ``\\widehat{R}``. See [^Margossian2024] for details.
"""
function rhat_nested(
    samples::AbstractArray{<:Union{Missing,Real}},
    superchain_ids::AbstractVector;
    kind::Symbol=:rank,
    split_chains::Int=2,
)
    ndims(samples) >= 2 || throw(
        ArgumentError(
            "`samples` must have at least 2 dimensions `(draws, chains[, parameters…])`"
        ),
    )
    chain_inds = _validate_superchain_ids(superchain_ids, size(samples, 2))
    if kind === :rank
        return _rhat_nested(Val(:rank), samples, chain_inds; split_chains)
    elseif kind === :bulk
        return _rhat_nested(Val(:bulk), samples, chain_inds; split_chains)
    elseif kind === :tail
        return _rhat_nested(Val(:tail), samples, chain_inds; split_chains)
    elseif kind === :basic
        return _rhat_nested(Val(:basic), samples, chain_inds; split_chains)
    else
        throw(ArgumentError("the `kind` `$kind` is not supported by `rhat_nested`"))
    end
end

function _validate_superchain_ids(superchain_ids, nchains)
    length(superchain_ids) == nchains || throw(
        DimensionMismatch(
            "`superchain_ids` has length $(length(superchain_ids)) but `samples` has $nchains chains",
        ),
    )
    _, chain_inds = unique_indices(superchain_ids)
    nsuperchains = length(chain_inds)
    nsuperchains >= 2 ||
        throw(ArgumentError("at least 2 superchains are required, got $nsuperchains"))
    allequal(length, chain_inds) ||
        throw(ArgumentError("all superchains must contain the same number of chains"))
    return reduce(hcat, chain_inds)
end

function _rhat_nested(
    ::Val{:basic},
    chains::AbstractArray{<:Union{Missing,Real}},
    chain_inds::AbstractMatrix{<:Integer};
    split_chains::Int=2,
)
    axes_out = _param_axes(chains)
    T = promote_type(eltype(chains), typeof(zero(eltype(chains)) / 1))
    rhat = similar(chains, T, axes_out)
    if T !== Missing
        _rhat_nested_basic!(rhat, chains, chain_inds; split_chains)
    end
    return _maybescalar(rhat)
end

function _rhat_nested(
    ::Val{:bulk},
    x::AbstractArray{<:Union{Missing,Real}},
    chain_inds::AbstractMatrix{<:Integer};
    kwargs...,
)
    return _rhat_nested(Val(:basic), _rank_normalize(x), chain_inds; kwargs...)
end

function _rhat_nested(
    ::Val{:tail},
    x::AbstractArray{<:Union{Missing,Real}},
    chain_inds::AbstractMatrix{<:Integer};
    kwargs...,
)
    return _rhat_nested(Val(:bulk), _fold_around_median(x), chain_inds; kwargs...)
end

function _rhat_nested(
    ::Val{:rank},
    x::AbstractArray{<:Union{Missing,Real}},
    chain_inds::AbstractMatrix{<:Integer};
    kwargs...,
)
    Rbulk = _rhat_nested(Val(:bulk), x, chain_inds; kwargs...)
    Rtail = _rhat_nested(Val(:tail), x, chain_inds; kwargs...)
    return map(max, Rbulk, Rtail)
end

function _rhat_nested_basic!(
    rhat::AbstractArray{T},
    chains::AbstractArray{<:Union{Missing,Real}},
    chain_inds::AbstractMatrix{<:Integer};
    split_chains::Int=2,
) where {T<:Union{Missing,Real}}
    # compute size of matrices (each chain may be split!)
    niter = size(chains, 1) ÷ split_chains
    nchains_per_superchain = size(chain_inds, 1) * split_chains
    nsuperchains = size(chain_inds, 2)

    # define caches for mean and variance
    chain_mean = Array{T}(undef, 1, nchains_per_superchain)
    chain_var = Array{T}(undef, nchains_per_superchain)
    superchain_mean = Array{T}(undef, nsuperchains)
    samples = Array{T}(undef, niter, nchains_per_superchain)

    # for each parameter
    for (i, chains_slice) in zip(eachindex(rhat), _eachparam(chains))
        # check that no values are missing
        if any(x -> x === missing, chains_slice)
            rhat[i] = missing
            continue
        end

        # estimate of within-superchain variance
        var_within_superchain = zero(T)

        # compute within-superchain quantities
        for (k, inds) in enumerate(eachcol(chain_inds))
            samples_in_superchain = view(chains_slice, :, inds)
            copyto_split!(samples, samples_in_superchain)

            # calculate mean of chains
            Statistics.mean!(chain_mean, samples)

            # calculuate superchain mean
            superchain_mean[k] = Statistics.mean(chain_mean)

            # calculate within-chain variance
            @inbounds for j in 1:nchains_per_superchain
                chain_var[j] = Statistics.var(
                    view(samples, :, j); mean=chain_mean[j], corrected=true
                )
            end
            Wk = Statistics.mean(chain_var)

            # calculate between-chain variance
            Bk = Statistics.var(chain_mean; corrected=(nchains_per_superchain > 1))

            var_within_superchain += Wk + Bk
        end
        var_within_superchain /= nsuperchains

        var_between_superchains = Statistics.var(superchain_mean; corrected=true)

        # estimate rhat
        rhat[i] = sqrt(1 + var_between_superchains / var_within_superchain)
    end

    return rhat
end
