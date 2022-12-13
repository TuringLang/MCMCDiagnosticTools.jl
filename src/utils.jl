"""
    indices_of_unique(x) -> Dict

Return a `Dict` whose keys are the unique elements of `x` and whose values are the
corresponding indices in `x`.
"""
function indices_of_unique(x)
    d = Dict{eltype(x), Vector{Int}}()
    for (i, xi) in enumerate(x)
        if haskey(d, xi)
            push!(d[xi], i)
        else
            d[xi] = [i]
        end
    end
    return d
end

