"""
    p_direction()

Compute the **Probability of Direction** (*pd*). It varies between `50%` and `100%` (*i.e.*, `0.5`
and `1`) and can be interpreted as the probability (expressed in percentage) that a parameter
(described by its posterior distribution) is strictly positive or negative (whichever is the most
probable). Although interpreted diffrerently, this index has been presented as statistically
related to the frequentist *p*-value.

# Examples
```jldoctest
julia> p_direction([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
0.6
```
"""
function p_direction(x::AbstractArray{<:Union{Missing,Real}})
    ntotal = 0
    npositive = 0
    nnegative = 0
    for xi in x
        if xi > 0
            npositive += 1
        elseif xi < 0
            nnegative += 1
        end
        ntotal += 1
    end
    return max(npositive, nnegative) / ntotal
end