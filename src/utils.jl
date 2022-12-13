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
