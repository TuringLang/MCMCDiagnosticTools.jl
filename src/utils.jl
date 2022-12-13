"""
    unique_indices(x) -> (unique, indices)

Return the results of `unique(collect(x))` along with the a vector of the same length whose
elements are the indices in `x` at which the corresponding unique element in `unique` is
found.
"""
function unique_indices(x)
    inds = eachindex(x)
    T = eltype(inds)
    ind_map = Dict{eltype(x),Vector{T}}()
    for i in inds
        xi = x[i]
        inds_xi = get!(ind_map, xi) do
            return T[]
        end
        push!(inds_xi, i)
    end
    unique = sort!(collect(keys(ind_map)))
    indices = [ind_map[xi] for xi in unique]
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
function split_chain_indices(c::AbstractVector{<:Int}, split::Int=2)
    cnew = similar(c)
    if split == 1
        copyto!(cnew, c)
        return cnew
    end
    chains, indices = unique_indices(c)
    chain_ind = 0
    for (chain, inds) in zip(chains, indices)
        ndraws_per_split, rem = divrem(length(inds), split)
        ilast = 0
        # here we can't use Iterators.partition because it's greedy. e.g. we can't partition
        # 4 items across 3 partitions because Iterators.partition(1:4, 1) == [[1], [2], [3]]
        # and Iterators.partition(1:4, 2) == [[1, 2], [3, 4]]. But we would want
        # [[1, 2], [3], [4]].
        for j in 1:split
            chain_ind += 1
            ndraws_this_split = ndraws_per_split + (j ≤ rem)
            i = ilast + 1
            ilast = i + ndraws_this_split - 1
            @views cnew[inds[i:ilast]] .= chain_ind
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
    rng::Random.AbstractRNG, groups::AbstractVector, frac::Real
)
    T = eltype(eachindex(groups))
    groups, indices = unique_indices(groups)
    inds1 = T[]
    inds2 = T[]
    for (group, inds) in zip(groups, indices)
        N = length(inds)
        N1 = round(Int, N * frac)
        ids = Random.randperm(rng, N)
        @views append!(inds1, inds[ids[1:N1]])
        @views append!(inds2, inds[ids[(N1 + 1):N]])
    end
    return inds1, inds2
end
