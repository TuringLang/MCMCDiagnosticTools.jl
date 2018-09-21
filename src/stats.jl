#################### Posterior Statistics ####################
function autocor(c::AbstractChains; 
                 lags::Vector=[1, 5, 10, 50],
                 relative::Bool=true)
    if relative
        lags *= step(c)
    elseif any(lags .% step(c) .!= 0)
        throw(ArgumentError("lags do not correspond to thinning interval"))
    end
    labels = map(x -> "Lag " * string(x), lags)
    
    (niter, nvar, nchain) = size(c.value)

    vals = zeros(nvar, length(lags), nchain)
    for k in 1:nchain
        for v in 1:nvar
            # skipping missing values
            # skipping should be done inside of autocor to ensure correct alignment
            # TODO: modify autocor to deal with missing values
            x = convert(Vector{Float64}, collect(skipmissing(c.value[:,v,k])))
            vals[v, :, k] = autocor(x, lags)
        end
    end
    ChainSummary(vals, c.names, labels, header(c))
end

function cor(c::AbstractChains)
    return ChainSummary(cor(combine(c)), c.names, c.names, header(c))
end

function changerate(c::AbstractChains)

    @assert !any(ismissing.(c.value)) "Change rate comp. doesn't support missing values."

    n, p, m = size(c.value)
    r = zeros(Float64, p, 1, 1)
    r_mv = 0.0
    delta = Array{Bool}(undef, p)
    for k in 1:m
        prev = c.value[1, :, k]
        for i in 2:n
            for j in 1:p
                x = c.value[i, j, k]
                dx = x != prev[j]
                r[j] += dx
                delta[j] = dx
                prev[j] = x
            end
            r_mv += any(delta)
        end
    end
    vals = round.([r; r_mv] / (m * (n - 1)), digits = 3)
    return ChainSummary(vals, [c.names; "Multivariate"], ["Change Rate"], header(c))
end

describe(c::AbstractChains; args...) = describe(stdout, c; args...)

function describe(io::IO, 
                  c::AbstractChains;
                  q = [0.025, 0.25, 0.5, 0.75, 0.975],
                  etype=:bm,
                  args...
                 )
    ps_stats = summarystats(c; etype=etype, args...)
    ps_quantiles = quantile(c, q=q)
    println(io, ps_stats.header)
    print(io, "Empirical Posterior Estimates:\n")
    show(io, ps_stats)
    print(io, "Quantiles:\n")
    show(io, ps_quantiles)

    return ps_stats, ps_quantiles
end

function hpd(x::Vector{T}; alpha::Real=0.05) where {T<:Real}
    n = length(x)
    m = max(1, ceil(Int, alpha * n))

    y = sort(x)
    a = y[1:m]
    b = y[(n - m + 1):n]
    _, i = findmin(b - a)

    return [a[i], b[i]]
end

function hpd(c::AbstractChains; alpha::Real=0.05)
    pct = first(showoff([100.0 * (1.0 - alpha)]))
    labels = ["$(pct)% Lower", "$(pct)% Upper"]
    vals = permutedims(
        mapslices(x -> hpd(collect(skipmissing(x)), alpha=alpha), c.value, dims = [1, 3]),
        [2, 1, 3]
    )
    return ChainSummary(vals, c.names, labels, header(c))
end

function quantile(c::AbstractChains; q::Vector=[0.025, 0.25, 0.5, 0.75, 0.975])
    labels = map(x -> string(100 * x) * "%", q)
    vals = permutedims(
        mapslices(x -> quantile(collect(skipmissing(x)), q), c.value, dims = [1, 3]),
        [2, 1, 3]
    )
    return ChainSummary(vals, c.names, labels, header(c))
end

function summarystats(c::AbstractChains; etype=:bm, args...)
    f = x -> [
            mean(x),
            std(x),
            sem(x),
            mcse(x, etype; args...)
    ]

    labels = ["Mean", "SD", "Naive SE", "MCSE", "ESS"]
    vals = permutedims(
        mapslices(x -> f(collect(skipmissing(x))), c.value, dims =  [1, 3]),
        [2, 1, 3]
    )
    stats = [vals  min.((vals[:, 2] ./ vals[:, 4]).^2, size(c.value, 1))]
    return ChainSummary(stats, c.names, labels, header(c))
end
