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
    copyto_split!(out::AbstractMatrix, x::AbstractMatrix)

Copy the elements of matrix `x` to matrix `out`, in which each column of `x` is split `n`
times.

The size of `x` must be `(split * m + d, n)`, and the size of `out` must be
`(m, n * split)`, where the integer `split` is the number of columns into which each column
in `x` will be split, and where integer `d < split` is the number of extra rows of `x` that
will be discarded. If `d > 0`, then a single row of `x` is discarded after each of the first
`d` splits.
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

    jout = 0
    @inbounds for j in 1:ncols_x
        i = 0
        for k in 1:nsplits
            jout += 1
            for iout in 1:nrows_out
                i += 1
                out[iout, jout] = x[i, j]
            end
            i += (k ≤ nrows_discard)
        end
    end

    return out
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
    _fold_around_median(x::AbstractArray{<:Any,3})

Compute the absolute deviation of `x` from `Statistics.median(x)`.
"""
_fold_around_median(data) = abs.(data .- Statistics.median(data; dims=(1, 2)))

"""
    _rank_normalize(x::AbstractArray{<:Any,3})

Rank-normalize the inputs `x` along the first 2 dimensions.

Rank-normalization proceeds by first ranking the inputs using "tied ranking"
and then transforming the ranks to normal quantiles so that the result is standard
normally distributed.
"""
function _rank_normalize(x::AbstractArray{<:Any,3})
    y = similar(x, float(eltype(x)))
    map(_rank_normalize!, eachslice(y; dims=3), eachslice(x; dims=3))
    return y
end
function _rank_normalize!(values, x)
    rank = StatsBase.tiedrank(x)
    _normal_quantiles_from_ranks!(values, rank)
    map!(StatsFuns.norminvcdf, values, values)
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
