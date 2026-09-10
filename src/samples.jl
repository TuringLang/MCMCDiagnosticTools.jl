# The public representations distinguish dense and ragged chains structurally, even when
# every chain in a vector happens to have the same length.
const _DiagnosticSamples = Union{
    AbstractArray{<:Union{Missing,Real}},
    AbstractVector{<:AbstractArray{<:Union{Missing,Real}}},
}

"""
    _Samples(samples)

Common internal representation of dense and ragged samples.

`data` is a matrix with shape `(samples, params)`. Dense input is reshaped without copying;
ragged input is concatenated along the draw dimension. Original chain lengths are recorded
separately, as a constant fill array for dense input and a vector for ragged input,
preserving their distinct correction rules.

`prototype` and `param_axes` retain the original array's output allocation behavior and
parameter axes independently of the storage used for computation and transformations.
"""
struct _Samples{A<:AbstractMatrix,L<:AbstractVector{Int},P<:AbstractArray,AX<:Tuple}
    data::A
    lengths::L
    prototype::P
    param_axes::AX
end
function _Samples(x::AbstractArray{<:Union{Missing,Real}})
    ndraws = size(x, 1)
    nchains = size(x, 2)
    param_axes = _param_axes(x)
    lengths = FillArrays.Fill(ndraws, nchains)
    data = reshape(x, ndraws * nchains, :)
    return _Samples(data, lengths, x, param_axes)
end
function _Samples(x::AbstractVector{<:AbstractArray})
    isempty(x) && throw(ArgumentError("at least one chain is required"))
    prototype = first(x)
    ndims(prototype) > 0 || throw(ArgumentError("chains must have a draw dimension"))
    param_axes = Base.tail(axes(prototype))
    for chain in x
        eltype(chain) <: Union{Missing,Real} ||
            throw(ArgumentError("chain elements must be real numbers or missing"))
        ndims(chain) > 0 || throw(ArgumentError("chains must have a draw dimension"))
        Base.tail(axes(chain)) == param_axes ||
            throw(DimensionMismatch("all chains must have the same parameter axes"))
    end
    lengths = collect(map(Base.Fix2(size, 1), x))
    nparams = prod(size(prototype)[2:end])
    x_mats = collect(map(chain -> reshape(chain, size(chain, 1), nparams), x))
    data = convert(Matrix, reduce(vcat, x_mats))
    return _Samples(data, lengths, prototype, param_axes)
end

# useful aliases
const _DenseSamples = _Samples{<:AbstractMatrix,<:FillArrays.Fill}
const _RaggedSamples = _Samples{<:AbstractMatrix,<:AbstractVector}

_with_data(x::_Samples, data) = _Samples(data, x.lengths, x.prototype, x.param_axes)
_eachparam(x::_Samples) = eachcol(x.data)
_similar_result(x::_Samples, ::Type{T}) where {T} = similar(x.prototype, T, x.param_axes)

# Restore parameter axes after a reduction over the pooled draw dimension.
function _restore_param_axes(x::_Samples, values::AbstractArray)
    result = _similar_result(x, eltype(values))
    copyto!(result, reshape(values, size(result)))
    return _maybescalar(result)
end

_min_split_length(x::_Samples, split::Int) = minimum(x.lengths) ÷ split

function _allocate_split_samples(x::_DenseSamples, ::Type{T}, split::Int) where {T}
    n = _min_split_length(x, split)
    # parameters containing missing values are skipped before copying into this workspace.
    S = Base.nonmissingtype(T)
    return Matrix{S}(undef, n, length(x.lengths) * split)
end
function _allocate_split_samples(x::_RaggedSamples, ::Type{T}, split::Int) where {T}
    # parameters containing missing values are skipped before copying into this workspace.
    S = Base.nonmissingtype(T)
    # Keep every draw by giving the first rem(n, split) pieces one extra draw.
    return [
        Matrix{S}(undef, n ÷ split + (k ≤ n % split), 1) for n in x.lengths for k in 1:split
    ]
end

# transformations

_rank_normalize(x::_Samples) = _with_data(x, _rank_normalize(x.data, 2))
_fold_around_median(x::_Samples) = _with_data(x, _fold_around_median(x.data, 2))
