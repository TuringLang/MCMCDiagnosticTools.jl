"""
    split_chains(data::AbstractArray{<:Any,3}, split::Int=2)

Split each chain in `data` of shape `(ndraws, nchains, nparams)` into `split` chains.

If `ndraws` is not divisible by `split`, the last `mod(ndraws, split)` iterations are
dropped. The result is a reshaped view of `data`.
"""
function split_chains(data::AbstractArray{<:Any,3}, split::Int=2)
    ndraws, nchains, nparams = size(data)
    ndraws_split, niter_drop = divrem(ndraws, split)
    nchains_split = nchains * split
    data_sub = @views data[begin:(end-niter_drop), :, :]
    return reshape(data_sub, ndraws_split, nchains_split, nparams)
end

"""
    rank_normalize(x::AbstractVector)
    rank_normalize(x::AbstractArray{<:Any,3}; dims=(1, 2))

Rank-normalize the inputs `x` along the dimensions `dims`.

Rank-normalization proceeds by first ranking the inputs using "tied ranking"
and then transforming the ranks to normal quantiles so that the result is standard
normally distributed.
"""
function rank_normalize(x::AbstractArray{<:Any,3}; dims=(1, 2))
    # TODO: can we avoid mapslices and the allocations here?
    return mapslices(x; dims=dims) do xi
        return reshape(rank_normalize(vec(xi)), size(xi))
    end
end
function rank_normalize(x::AbstractVector)
    values = similar(x, float(eltype(x)))
    rank_normalize!(values, x)
    return values
end
function rank_normalize!(values, x)
    rank = StatsBase.tiedrank(x)
    _normal_quantiles_from_ranks!(values, rank)
    values .= StatsFuns.norminvcdf.(values)
    return values
end

