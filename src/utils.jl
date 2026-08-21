const _RaggedSamples = AbstractVector{<:AbstractArray{<:Union{Missing,Real}}}
const _DiagnosticSamples = Union{AbstractArray{<:Union{Missing,Real}},_RaggedSamples}
const _ArrayOfArrays = AbstractVector{<:AbstractArray}

"""
    copyto_split!(out::AbstractMatrix, x::AbstractMatrix)

Copy the elements of matrix `x` to matrix `out`, in which each column of `x` is split across
multiple columns of `out`.

To split each column of `x` into `split` columns, where the size of `x` is `(m, n)`, the
size of `out` must be `(m ÷ split, n * split)`.

If `d = rem(m, split) > 0`, so that `m` is not evenly divisible by `split`, then a single
row of `x` is discarded after each of the first `d` splits for each column.
"""
function copyto_split!(out::AbstractMatrix, x::AbstractMatrix)
    # check dimensions
    nrows_out, ncols_out = size(out)
    nrows_x, ncols_x = size(x)
    nsplits, ncols_extra = divrem(ncols_out, ncols_x)
    ncols_extra == 0 || throw(
        DimensionMismatch(
            "the output matrix must have an integer multiple of the number of columns evenly divisible by the those of the input matrix",
        ),
    )
    nrows_out2, nrows_discard = divrem(nrows_x, nsplits)
    nrows_out == nrows_out2 || throw(
        DimensionMismatch(
            "the output matrix must have $nsplits times as many rows as as the input matrix",
        ),
    )
    if nrows_discard > 0
        offset = firstindex(x)
        offset_out = firstindex(out)
        for _ in 1:ncols_x, k in 1:nsplits
            copyto!(out, offset_out, x, offset, nrows_out)
            offset += nrows_out + (k ≤ nrows_discard)
            offset_out += nrows_out
        end
    else
        copyto!(out, reshape(x, nrows_out, ncols_out))
    end
    return out
end

function copyto_split!(
    out::AbstractVector{<:AbstractVector}, x::AbstractVector{<:AbstractVector}
)
    nsplits, n_extra = divrem(length(out), length(x))
    n_extra == 0 || throw(
        DimensionMismatch(
            "the output must contain an integer multiple of the number of input chains"
        ),
    )
    out_index = firstindex(out)
    for chain in x
        nrows_out, nrows_discard = divrem(length(chain), nsplits)
        offset = firstindex(chain)
        for k in 1:nsplits
            out_chain = out[out_index]
            length(out_chain) == nrows_out || throw(
                DimensionMismatch(
                    "each output chain must contain the number of draws in its input chain divided by the number of splits",
                ),
            )
            copyto!(out_chain, firstindex(out_chain), chain, offset, nrows_out)
            offset += nrows_out + (k ≤ nrows_discard)
            out_index += 1
        end
    end
    return out
end

"""
    unique_indices(x) -> (unique, indices)

Return the results of `unique(collect(x))` along with the a vector of the same length whose
elements are the indices in `x` at which the corresponding unique element in `unique` is
found.
"""
function unique_indices(x)
    inds = eachindex(x)
    T = eltype(inds)
    ind_map = DataStructures.SortedDict{eltype(x),Vector{T}}()
    for i in inds
        xi = x[i]
        inds_xi = get!(ind_map, xi) do
            return T[]
        end
        push!(inds_xi, i)
    end
    unique = collect(keys(ind_map))
    indices = collect(values(ind_map))
    return unique, indices
end

"""
    split_chain_indices(
        chain_inds::AbstractVector{Int},
        split::Int=2,
    ) -> AbstractVector{Int}

Split each chain in `chain_inds` into `split` chains.

For each chain in `chain_inds`, all entries are assumed to correspond to draws that have
been ordered by iteration number. The result is a vector of the same length as `chain_inds`
where each entry is the new index of the chain that the corresponding draw belongs to.
"""
function split_chain_indices(c::AbstractVector{Int}, split::Int=2)
    cnew = similar(c)
    if split == 1
        copyto!(cnew, c)
        return cnew
    end
    _, indices = unique_indices(c)
    chain_ind = 1
    for inds in indices
        ndraws_per_split, rem = divrem(length(inds), split)
        # here we can't use Iterators.partition because it's greedy. e.g. we can't partition
        # 4 items across 3 partitions because Iterators.partition(1:4, 1) == [[1], [2], [3]]
        # and Iterators.partition(1:4, 2) == [[1, 2], [3, 4]]. But we would want
        # [[1, 2], [3], [4]].
        i = j = 0
        ndraws_this_split = ndraws_per_split + (j < rem)
        for ind in inds
            cnew[ind] = chain_ind
            if (i += 1) == ndraws_this_split
                i = 0
                j += 1
                ndraws_this_split = ndraws_per_split + (j < rem)
                chain_ind += 1
            end
        end
    end
    return cnew
end

"""
    shuffle_split_stratified(
        rng::Random.AbstractRNG,
        group_ids::AbstractVector,
        frac::Real,
    ) -> (inds1, inds2)

Randomly split the indices of `group_ids` into two groups, where `frac` indices from each
group are in `inds1` and the remainder are in `inds2`.

This is used, for example, to split data into training and test data while preserving the
class balances.
"""
function shuffle_split_stratified(
    rng::Random.AbstractRNG, group_ids::AbstractVector, frac::Real
)
    _, indices = unique_indices(group_ids)
    T = eltype(eltype(indices))
    N1_tot = sum(x -> round(Int, length(x) * frac), indices)
    N2_tot = length(group_ids) - N1_tot
    inds1 = Vector{T}(undef, N1_tot)
    inds2 = Vector{T}(undef, N2_tot)
    items_in_1 = items_in_2 = 0
    for inds in indices
        N = length(inds)
        N1 = round(Int, N * frac)
        N2 = N - N1
        Random.shuffle!(rng, inds)
        copyto!(inds1, items_in_1 + 1, inds, 1, N1)
        copyto!(inds2, items_in_2 + 1, inds, N1 + 1, N2)
        items_in_1 += N1
        items_in_2 += N2
    end
    return inds1, inds2
end

"""
    _fold_around_median(x::AbstractArray)

Compute the absolute deviation of `x` from `Statistics.median(x)`.
"""
function _fold_around_median(x::_DiagnosticSamples)
    T0 = _sample_eltype(x)
    T = promote_type(T0, typeof(zero(T0) / 1))
    y = _similar_samples(x, T)
    # avoid using the `dims` keyword for median because it
    # - can error for Union{Missing,Real} (https://github.com/JuliaStats/Statistics.jl/issues/8)
    # - is type-unstable (https://github.com/JuliaStats/Statistics.jl/issues/39)
    for (xi, yi) in zip(_eachparam(x), _eachparam(y))
        _fold_around_median!(yi, xi)
    end
    return y
end
function _fold_around_median!(y::AbstractArray{<:Union{Missing,Real}}, x)
    y .= abs.(x .- Statistics.median(vec(x)))
    return y
end
function _fold_around_median!(y::_ArrayOfArrays, x::_ArrayOfArrays)
    median = Statistics.median(_pool(x))
    for (yi, xi) in zip(y, x)
        yi .= abs.(xi .- median)
    end
    return y
end

"""
    _rank_normalize(x::AbstractArray)

Rank-normalize the inputs `x` along the sample dimensions.

Rank-normalization proceeds by first ranking the inputs using "tied ranking"
and then transforming the ranks to normal quantiles so that the result is standard
normally distributed.
"""
function _rank_normalize(x::AbstractArray)
    T0 = _sample_eltype(x)
    T = promote_type(T0, typeof(zero(T0) / 1))
    y = _similar_samples(x, T)
    map(_rank_normalize!, _eachparam(y), _eachparam(x))
    return y
end
function _rank_normalize!(values, x)
    if any(ismissing, x)
        fill!(values, missing)
        return values
    end
    rank = StatsBase.tiedrank(x)
    _normal_quantiles_from_ranks!(values, rank)
    map!(StatsFuns.norminvcdf, values, values)
    return values
end
function _rank_normalize!(values::_ArrayOfArrays, x::_ArrayOfArrays)
    if any(chain -> any(ismissing, chain), x)
        foreach(Base.Fix2(fill!, missing), values)
        return values
    end
    pooled = _pool(x)
    rank = StatsBase.tiedrank(pooled)
    _normal_quantiles_from_ranks!(rank, rank)
    map!(StatsFuns.norminvcdf, rank, rank)
    offset = firstindex(rank)
    for chain in values
        n = length(chain)
        copyto!(chain, firstindex(chain), rank, offset, n)
        offset += n
    end
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

# utilities for supporting input arrays with an arbitrary number of dimensions

_validate_samples(::AbstractArray{<:Union{Missing,Real}}) = nothing
function _validate_samples(chains::_RaggedSamples)
    isempty(chains) && throw(ArgumentError("`samples` must contain at least one chain"))
    first_chain = first(chains)
    ndims(first_chain) > 0 ||
        throw(DimensionMismatch("each chain must have at least one draw dimension"))
    size(first_chain, 1) > 0 ||
        throw(ArgumentError("each chain must contain at least one draw"))
    param_axes = Base.tail(axes(first_chain))
    for chain in Iterators.drop(chains, 1)
        size(chain, 1) > 0 ||
            throw(ArgumentError("each chain must contain at least one draw"))
        Base.tail(axes(chain)) == param_axes ||
            throw(DimensionMismatch("all chains must have identical parameter axes"))
    end
    return nothing
end

_sample_eltype(x::AbstractArray{<:Union{Missing,Real}}) = eltype(x)
function _sample_eltype(chains::_ArrayOfArrays)
    return mapreduce(eltype, promote_type, chains)
end

function _promote_sample_eltype(x::AbstractArray{<:Union{Missing,Real}}, values...)
    return Base.promote_eltype(x, values...)
end
function _promote_sample_eltype(chains::_ArrayOfArrays, values...)
    return promote_type(_sample_eltype(chains), map(typeof, values)...)
end

_similar_samples(x::AbstractArray{<:Union{Missing,Real}}, T) = similar(x, T)
function _similar_samples(chains::_ArrayOfArrays, T)
    return map(chain -> similar(chain, T), chains)
end
_similar_samples(x::_DiagnosticSamples) = _similar_samples(x, _sample_eltype(x))

_pool(x::AbstractArray{<:Union{Missing,Real}}) = vec(x)
function _pool(chains::_ArrayOfArrays)
    T = _sample_eltype(chains)
    pooled = Vector{T}(undef, sum(length, chains))
    offset = firstindex(pooled)
    for chain in chains
        n = length(chain)
        copyto!(pooled, offset, chain, firstindex(chain), n)
        offset += n
    end
    return pooled
end

_any_missing(x::AbstractArray{<:Union{Missing,Real}}) = any(ismissing, x)
_any_missing(chains::_ArrayOfArrays) = any(chain -> any(ismissing, chain), chains)

function _fill_samples!(x::AbstractArray{<:Union{Missing,Real}}, value)
    fill!(x, value)
    return x
end
function _fill_samples!(chains::_ArrayOfArrays, value)
    foreach(Base.Fix2(fill!, value), chains)
    return chains
end

function _threshold!(y::AbstractArray{<:Union{Missing,Real}}, x, threshold)
    y .= x .≤ threshold
    return y
end
function _threshold!(y::_ArrayOfArrays, x::_ArrayOfArrays, threshold)
    for (yi, xi) in zip(y, x)
        yi .= xi .≤ threshold
    end
    return y
end

function _squared_deviations(x::AbstractArray{<:Union{Missing,Real}})
    dims = _sample_dims(x)
    return (x .- Statistics.mean(x; dims=dims)) .^ 2
end
function _squared_deviations(chains::_ArrayOfArrays)
    T0 = _sample_eltype(chains)
    T = promote_type(T0, typeof(zero(T0) / 1))
    y = _similar_samples(chains, T)
    for (xi, yi) in zip(_eachparam(chains), _eachparam(y))
        mean = Statistics.mean(_pool(xi))
        for (xij, yij) in zip(xi, yi)
            @. yij = abs2(xij - mean)
        end
    end
    return y
end

function _similar_params(x::AbstractArray{<:Union{Missing,Real}}, T)
    return similar(x, T, _param_axes(x))
end
function _similar_params(chains::_ArrayOfArrays, T)
    return similar(first(chains), T, _param_axes(chains))
end

_min_split_draws(x::AbstractArray{<:Union{Missing,Real}}, split::Int) = size(x, 1) ÷ split
function _min_split_draws(chains::_ArrayOfArrays, split::Int)
    return minimum(chain -> size(chain, 1) ÷ split, chains)
end

_nsplit_chains(x::AbstractArray{<:Union{Missing,Real}}, split::Int) = split * size(x, 2)
_nsplit_chains(chains::_ArrayOfArrays, split::Int) = split * length(chains)

function _total_split_draws(x::AbstractArray{<:Union{Missing,Real}}, split::Int)
    return _min_split_draws(x, split) * _nsplit_chains(x, split)
end
function _total_split_draws(chains::_ArrayOfArrays, split::Int)
    return sum(chain -> (size(chain, 1) ÷ split) * split, chains)
end

function _allocate_split(x::AbstractArray{<:Union{Missing,Real}}, T, split::Int)
    return Matrix{T}(undef, _min_split_draws(x, split), _nsplit_chains(x, split))
end
function _allocate_split(chains::_ArrayOfArrays, T, split::Int)
    samples = Vector{Vector{T}}(undef, _nsplit_chains(chains, split))
    i = 1
    for chain in chains
        niter = size(chain, 1) ÷ split
        for _ in 1:split
            samples[i] = Vector{T}(undef, niter)
            i += 1
        end
    end
    return samples
end

function _mean_var!(chain_mean, chain_var, samples::AbstractMatrix)
    Statistics.mean!(chain_mean, samples)
    @inbounds for j in axes(samples, 2)
        chain_var[j] = Statistics.var(
            view(samples, :, j); mean=chain_mean[j], corrected=true
        )
    end
    return nothing
end
function _mean_var!(chain_mean, chain_var, samples::AbstractVector{<:AbstractVector})
    @inbounds for j in eachindex(samples, chain_var)
        chain = samples[j]
        chain_mean[j] = Statistics.mean(chain)
        chain_var[j] = Statistics.var(chain; mean=chain_mean[j], corrected=true)
    end
    return nothing
end

function _center!(samples::AbstractMatrix, chain_mean)
    samples .-= chain_mean
    return samples
end
function _center!(samples::AbstractVector{<:AbstractVector}, chain_mean)
    for (chain, mean) in zip(samples, chain_mean)
        chain .-= mean
    end
    return samples
end

_nchains(samples::AbstractMatrix) = size(samples, 2)
_nchains(samples::AbstractVector{<:AbstractVector}) = length(samples)
_min_draws(samples::AbstractMatrix) = size(samples, 1)
_min_draws(samples::AbstractVector{<:AbstractVector}) = minimum(length, samples)
_max_draws(samples::AbstractMatrix) = size(samples, 1)
_max_draws(samples::AbstractVector{<:AbstractVector}) = maximum(length, samples)

function _copyto_fft!(samples_cache, samples::AbstractMatrix)
    niter, nchains = size(samples)
    n = size(samples_cache, 1)
    T = eltype(samples_cache)
    @inbounds for j in 1:nchains
        for i in 1:niter
            samples_cache[i, j] = samples[i, j]
        end
        for i in (niter + 1):n
            samples_cache[i, j] = zero(T)
        end
    end
    return samples_cache
end
function _copyto_fft!(samples_cache, samples::AbstractVector{<:AbstractVector})
    n = size(samples_cache, 1)
    T = eltype(samples_cache)
    @inbounds for j in eachindex(samples)
        chain = samples[j]
        niter = length(chain)
        for i in 1:niter
            samples_cache[i, j] = chain[i]
        end
        for i in (niter + 1):n
            samples_cache[i, j] = zero(T)
        end
    end
    return samples_cache
end

function _rhat_correction(x::AbstractArray{<:Union{Missing,Real}}, split::Int)
    niter = _min_split_draws(x, split)
    return (niter - 1)//niter
end
_rhat_correction(::_ArrayOfArrays, ::Int) = 1

_sample_dims(x::AbstractArray) = ntuple(identity, min(2, ndims(x)))

_param_dims(x::AbstractArray) = ntuple(i -> i + 2, max(0, ndims(x) - 2))

_param_axes(x::AbstractArray) = map(Base.Fix1(axes, x), _param_dims(x))

_param_axes(chains::_ArrayOfArrays) = Base.tail(axes(first(chains)))

function _params_array(x::AbstractArray, param_dim::Int=3)
    param_dim > 0 || throw(ArgumentError("param_dim must be positive"))
    sample_sizes = ntuple(Base.Fix1(size, x), param_dim - 1)
    return reshape(x, sample_sizes..., :)
end

function _eachparam(x::AbstractArray, param_dim::Int=3)
    return eachslice(_params_array(x, param_dim); dims=param_dim)
end
function _eachparam(chains::_ArrayOfArrays, param_dim::Int=2)
    param_dim == 2 || throw(ArgumentError("ragged chains only support `param_dim=2`"))
    param_axes = _param_axes(chains)
    return (
        map(chain -> view(chain, axes(chain, 1), Tuple(I)...), chains) for
        I in CartesianIndices(param_axes)
    )
end

# convert 0-dimensional arrays to scalars
_maybescalar(x::AbstractArray{<:Any,0}) = x[]
_maybescalar(x::AbstractArray) = x
