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
    data_sub = @views data[begin:(end - niter_drop), :, :]
    return reshape(data_sub, ndraws_split, nchains_split, nparams)
end

_sample_dims(data::AbstractVector) = Colon()
_sample_dims(data::AbstractArray{<:Any,3}) = (1, 2)

"""
    fold([f,] x::AbstractVector)
    fold([f,] x::AbstractArray{<:Any,3}; dims=(1, 2))

Compute the absolute deviation of `x` from `f(x)`, where `f` defaults to `Statistics.median`.

`f` is generally a measure of central tendency. `dims` are the dimensions over which the
estimator `f` reduces and are passed as kwargs to `f`.
"""
fold(f, data; dims=_sample_dims(data)) = abs.(data .- f(data; dims=dims))
fold(data; kwargs...) = fold(Statistics.median, data; kwargs...)

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

# transform the ranks to quantiles of a standard normal distribution applying the
# "α-β correction" recommended in Eq 6.10.3 of
# Blom. Statistical Estimates and Transformed Beta-Variables. Wiley; New York, 1958
function _normal_quantiles_from_ranks!(q, r; α=3//8)
    n = length(r)
    q .= (r .- α) ./ (n - 2α + 1)
    return q
end

"""
    expectand_proxy(f, x; dims=:)

Compute an expectand `z` such that ``\\textrm{mean-ESS}(z) ≈ \\textrm{f-ESS}(x)``.

`f` should be a function that reduces a vector to a scalar or alternatively takes a `dims`
keyword that specifies the sample dimensions of `x`, that is, the draw and chain dimensions.

If no proxy expectand for `f` is known, `nothing` is returned.
"""
expectand_proxy(f, x; kwargs...) = nothing
expectand_proxy(::typeof(Statistics.mean), x; kwargs...) = x
function expectand_proxy(::typeof(Statistics.median), x; dims=:)
    return x .≤ Statistics.median(x; dims=dims)
end
function expectand_proxy(::typeof(Statistics.std), x; dims=:)
    return (x .- Statistics.mean(x; dims=dims)).^2
end
function expectand_proxy(::typeof(StatsBase.mad), x; dims=:)
    x_folded = fold(Statistics.median, x; dims=dims)
    return expectand_proxy(Statistics.median, x_folded; dims=dims)
end
# currently quantile does not support a dims keyword argument
function expectand_proxy(f::Base.Fix2{typeof(Statistics.quantile),<:Real}, x; dims=:)
    dims isa Colon && return x .≤ f(vec(x))
    return x .≤ mapslices(f ∘ vec, x; dims=dims)
end
