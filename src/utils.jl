"""
    indices_of_unique(x) -> Dict

Return a `Dict` whose keys are the unique elements of `x` and whose values are the
corresponding indices in `x`.
"""
function indices_of_unique(x)
    d = Dict{eltype(x),Vector{Int}}()
    for (i, xi) in enumerate(x)
        if haskey(d, xi)
            push!(d[xi], i)
        else
            d[xi] = [i]
        end
    end
    return d
end

"""
    split_chain_indices(
        chain_inds::AbstractVector{Int},
        nsplit::Int=2,
    ) -> AbstractVector{Int}

Split each chain in `chain_inds` into `nsplit` chains.

For each chain in `chain_inds`, all entries are assumed to correspond to draws that have
been ordered by iteration number. The result is a vector of the same length as `chain_inds`
where each entry is the new index of the chain that the corresponding draw belongs to.
"""
function split_chain_indices(c::AbstractVector{<:Int}, nsplit::Int=2)
    cnew = similar(c)
    if nsplit == 1
        copyto!(cnew, c)
        return cnew
    end
    chain_indices = indices_of_unique(c)
    chain_ind = 0
    for chain in sort(collect(keys(chain_indices)))
        inds = chain_indices[chain]
        ndraws_per_split, rem = divrem(length(inds), nsplit)
        ilast = 0
        for j in 1:nsplit
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
    inds1 = Int[]
    inds2 = Int[]
    group_indices = indices_of_unique(groups)
    for group in keys(group_indices)
        inds = group_indices[group]
        N = length(inds)
        N1 = round(Int, N * frac)
        ids = Random.randperm(rng, N)
        @views append!(inds1, inds[ids[1:N1]])
        @views append!(inds2, inds[ids[(N1 + 1):N]])
    end
    return inds1, inds2
end
