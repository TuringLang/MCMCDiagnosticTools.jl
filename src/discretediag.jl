function update_hangartner_temp_vars!(u::Matrix{Int}, X::Matrix{Int}, t::Int)
    d = size(X, 2)

    for j in 1:d
        u[X[t, j], j] += 1
    end
end

function hangartner_inner(Y::AbstractMatrix, m::Int)
    ## setup temp vars
    n, d = size(Y)

    # Count for each category in each chain
    u = zeros(Int, m, d)
    v = zeros(Int, m, d)

    for t in 1:n
        # fill out temp vars
        update_hangartner_temp_vars!(u, Y, t)
    end
    phia, chi_stat, m_tot = weiss_sub(u, v, n)

    return (n * sum(chi_stat), m_tot)
end

"""
    weiss(X::AbstractMatrix)

Assess the convergence of the MCMC chains with the Weiss procedure.

It computes ``\\frac{X^2}{c}`` and evaluates a p-value from the ``\\chi^2`` distribution with ``(|R| − 1)(s − 1)`` degrees of freedom.
"""
function weiss(X::AbstractMatrix)
    ## number of iterations, number of chains
    n, d = size(X)

    ## mapping of values to integers
    v_dict = Dict{eltype(X),Int}()

    ## max unique categories
    mc = map(c -> length(unique(X[:, c])), 1:d)
    m = length(unique(X))

    ## counter of number of unique values in each chain
    r0 = 0

    ## Count for each category in each chain
    u = zeros(Int, m, d)

    ## Number of times a category did not transition in each chain
    v = zeros(Int, m, d)

    for t in 1:n
        for c in 1:d
            if !(X[t, c] in keys(v_dict))
                r0 += 1
                v_dict[X[t, c]] = r0
            end
            idx1 = v_dict[X[t, c]]
            u[idx1, c] += 1

            if t > 1
                if X[t - 1, c] == X[t, c]
                    v[idx1, c] += 1
                end
            end
        end
    end
    phia, chi_stat, m_tot = weiss_sub(u, v, n)
    ca = (1 + phia) / (1 - phia)
    stat = (n / ca) * sum(chi_stat)
    pval = NaN
    if ((m_tot - 1) * (d - 1)) >= 1
        pval = Distributions.ccdf(Distributions.Chisq((m_tot - 1) * (d - 1)), stat)
    end

    return (stat, m_tot, pval, ca)
end

function weiss_sub(u::Matrix{Int}, v::Matrix{Int}, t::Int)
    m, d = size(u)
    nt = 0.0
    dt = 0.0
    m_tot = 0

    mp = zeros(Float64, m, d)
    ma = zeros(Float64, m)
    phia = 0.0
    ca = 0.0
    df = 0.0

    chi_stat = zeros(Float64, d)

    for j in 1:m
        p1 = 0.0
        p2 = 0.0
        for l in 1:d
            #aggregate
            p1 += v[j, l] / (d * (t - 1))
            p2 += u[j, l] / (d * t)

            #per chain
            mp[j, l] = u[j, l] / t
            ma[j] += u[j, l] / (d * t)
        end
        nt += p1
        dt += p2^2

        if ma[j] > 0
            m_tot += 1
            for l in 1:d
                chi_stat[l] += (mp[j, l] - ma[j])^2 / ma[j]
            end
        end
    end
    phia = 1.0 + (1.0 / t) - ((1 - nt) / (1 - dt))
    phia = min(max(phia, 0.0), 1.0 - eps())
    return (phia, chi_stat, m_tot)
end

function update_billingsley_temp_vars!(f::Array{Int,3}, X::Matrix{Int}, t::Int)
    d = size(X, 2)
    for j in 1:d
        if t > 1
            f[X[t - 1, j], X[t, j], j] += 1
        end
    end
end

function billingsley_sub(f::Array{Int,3})
    df = 0.0
    stat = 0.0

    m, d = size(f)[2:3]

    # marginal transitions, i.e.
    # number of transitions from each category
    mf = mapslices(sum, f; dims=[2])

    # For each category, number of chains for which
    # that category was present
    A = vec(mapslices((x) -> sum(x .> 0), mf; dims=[3]))

    # For each category, number of categories it
    # transitioned to
    B = vec(mapslices((x) -> sum(x .> 0), mapslices(sum, f; dims=[3]); dims=[2]))

    # transition probabilities in each chain
    P = f ./ mf

    # transition probabilities
    mP = (mapslices(sum, f; dims=[3]) ./ mapslices(sum, mf; dims=[3]))
    mP = reshape(mP, size(mP)[1:2])

    idx = findall((A .* B) .> 0)
    for j in idx
        #df for billingsley
        df += (A[j] - 1) * (B[j] - 1)

        #billingsley
        for k in idx
            if (mP[j, k] > 0.0)
                for l in 1:d
                    if mf[j, 1, l] > 0 && isfinite(P[j, k, l])
                        stat += mf[j, 1, l] * (P[j, k, l] - mP[j, k])^2 / mP[j, k]
                    end
                end
            end
        end
    end
    return (stat, df, mP)
end

function bd_inner(Y::AbstractMatrix, m::Int)
    num_iters, num_chains = size(Y)
    # Transition matrix for each chain
    f = zeros(Int, m, m, num_chains)

    for t in 1:num_iters
        # fill out temp vars
        update_billingsley_temp_vars!(f, Y, t)
    end
    return billingsley_sub(f)
end

@doc raw"""
    simulate_DAR1!(X::Matrix{Int}, phi::Float64, sampler)

Simulate a DAR(1) model independently in each column of `X`.

The DAR(1) model ``(X_t)_{t \geq 0}`` is defined by
```math
X_t = \alpha_t X_{t-1} + (1 - \alpha_t) \epsilon_{t-1},
```
where
```math
\begin{aligned}
X_1 \sim \text{sampler}, \\
\alpha_t \sim \operatorname{Bernoulli}(\phi), \\
\epsilon_{t-1} \sim \text{sampler},
\end{aligned}
```
are independent random variables.
"""
function simulate_DAR1!(X::Matrix{Int}, phi::Float64, sampler)
    n = size(X, 1)
    n > 0 || error("output matrix must be non-empty")

    # for all simulations
    @inbounds for j in axes(X, 2)
        # sample first value from categorical distribution with probabilities `prob`
        X[1, j] = rand(sampler)

        for t in 2:n
            # compute next value
            X[t, j] = if rand() <= phi
                # copy previous value with probability `phi`
                X[t - 1, j]
            else
                # sample value with probability `1-phi`
                rand(sampler)
            end
        end
    end

    return X
end

function simulate_MC(N::Int, P::Matrix{Float64})
    X = zeros(Int, N)
    n, m = size(P)
    X[1] = StatsBase.sample(1:n)
    for i in 2:N
        X[i] = StatsBase.wsample(1:n, vec(P[X[i - 1], :]))
    end
    return X
end

function diag_all(
    X::AbstractMatrix, method::Symbol, nsim::Int, start_iter::Int, step_size::Int
)

    ## number of iterations, number of chains
    n, d = size(X)

    ## mapping of values to integers
    v_dict = Dict{eltype(X),Int}()

    ## max unique categories
    mc = map(c -> length(unique(X[:, c])), 1:d)
    m = length(unique(X))

    ## counter of number of unique values in each chain
    r0 = 0

    ## Count for each category in each chain
    u = zeros(Int, m, d)

    ## Number of times a category did not transition in each chain
    v = zeros(Int, m, d)

    ## transition matrix for each chain
    f = zeros(Int, m, m, d)

    length_result = length(start_iter:step_size:n)
    result = (
        stat=Vector{Float64}(undef, length_result),
        df=Vector{Float64}(undef, length_result),
        pvalue=Vector{Float64}(undef, length_result),
    )
    result_iter = 1
    for t in 1:n
        for c in 1:d
            if !(X[t, c] in keys(v_dict))
                r0 += 1
                v_dict[X[t, c]] = r0
            end
            idx1 = v_dict[X[t, c]]
            u[idx1, c] += 1

            if t > 1
                idx2 = v_dict[X[t - 1, c]]
                f[idx1, idx2, c] += 1

                if X[t - 1, c] == X[t, c]
                    v[idx1, c] += 1
                end
            end
        end

        if ((t >= start_iter) && (((t - start_iter) % step_size) == 0))
            phia, chi_stat, m_tot = weiss_sub(u, v, t)
            hot_stat, df, mP = billingsley_sub(f)

            phat = mapslices(sum, u; dims=[2])[:, 1] / sum(mapslices(sum, u; dims=[2]))
            ca = (1 + phia) / (1 - phia)
            stat = NaN
            pval = NaN
            df0 = NaN

            if method == :hangartner
                stat = t * sum(chi_stat)
                df0 = (m - 1) * (d - 1)
                if m > 1 && !isnan(stat)
                    pval =
                        Distributions.ccdf(Distributions.Chisq(df0), stat)
                end
            elseif method == :weiss
                stat = (t / ca) * sum(chi_stat)
                df0 = (m - 1) * (d - 1)
                pval = NaN
                if m > 1 && !isnan(stat)
                    pval =
                        Distributions.ccdf(Distributions.Chisq(df0), stat)
                end
            elseif method == :DARBOOT
                stat = t * sum(chi_stat)
                sampler_phat = Distributions.sampler(Distributions.Categorical(phat))
                bstats = zeros(nsim)
                Y = Matrix{Int}(undef, t, d)
                for b in 1:nsim
                    simulate_DAR1!(Y, phia, sampler_phat)
                    s = hangartner_inner(Y, m)[1]
                    @inbounds bstats[b] = s
                end
                non_nan_bstats = filter(!isnan, bstats)
                df0 = Statistics.mean(non_nan_bstats)
                pval = Statistics.mean(stat <= x for x in non_nan_bstats)
            elseif method == :MCBOOT
                bstats = zeros(Float64, nsim)
                for b in 1:nsim
                    Y = reduce(hcat, [simulate_MC(t, mP) for j in 1:d])
                    s = hangartner_inner(Y, m)[1]
                    bstats[b] = s
                end
                non_nan_bstats = filter(!isnan, bstats)
                df0 = Statistics.mean(non_nan_bstats)
                pval = Statistics.mean(stat <= x for x in non_nan_bstats)
            elseif method == :billingsley
                stat = hot_stat
                df0 = df
                if df > 0 && !isnan(hot_stat)
                    pval = Distributions.ccdf(Distributions.Chisq(df), hot_stat)
                end
            elseif method == :billingsleyBOOT
                stat = hot_stat
                bstats = zeros(Float64, nsim)
                for b in 1:nsim
                    Y = reduce(hcat, [simulate_MC(t, mP) for j in 1:d])
                    (s, sd) = bd_inner(Y, m)[1:2]
                    bstats[b] = s / sd
                end
                non_nan_bstats = filter(!isnan, bstats)
                df0 = Statistics.mean(non_nan_bstats)
                statodf = stat / df
                pval = Statistics.mean(statodf <= x for x in non_nan_bstats)
            else
                error("Unexpected")
            end
            result.stat[result_iter] = stat
            result.df[result_iter] = df0
            result.pvalue[result_iter] = pval
            result_iter += 1
        end
    end
    return result
end

function discretediag_sub(
    c::AbstractArray{<:Real,3},
    frac::Real,
    method::Symbol,
    nsim::Int,
    start_iter::Int,
    step_size::Int,
)
    num_iters, num_vars, num_chains = size(c)

    ## Between-chain diagnostic
    length_results = length(start_iter:step_size:num_iters)
    plot_vals_stat = Matrix{Float64}(undef, length_results, num_vars)
    plot_vals_pval = Matrix{Float64}(undef, length_results, num_vars)
    between_chain_vals = (
        stat=Vector{Float64}(undef, num_vars),
        df=Vector{Float64}(undef, num_vars),
        pvalue=Vector{Float64}(undef, num_vars),
    )
    for j in 1:num_vars
        X = convert(AbstractMatrix{Int}, c[:, j, :])
        result = diag_all(X, method, nsim, start_iter, step_size)

        plot_vals_stat[:, j] .= result.stat ./ result.df
        plot_vals_pval[:, j] .= result.pvalue

        between_chain_vals.stat[j] = result.stat[end]
        between_chain_vals.df[j] = result.df[end]
        between_chain_vals.pvalue[j] = result.pvalue[end]
    end

    ## Within-chain diagnostic
    within_chain_vals = (
        stat=Matrix{Float64}(undef, num_vars, num_chains),
        df=Matrix{Float64}(undef, num_vars, num_chains),
        pvalue=Matrix{Float64}(undef, num_vars, num_chains),
    )
    for k in 1:num_chains
        for j in 1:num_vars
            x = convert(AbstractVector{Int}, c[:, j, k])

            idx1 = 1:round(Int, frac * num_iters)
            idx2 = round(Int, num_iters - frac * num_iters + 1):num_iters
            x1 = x[idx1]
            x2 = x[idx2]
            n_min = min(length(x1), length(x2))
            Y = [x1[1:n_min] x2[(end - n_min + 1):end]]

            result = diag_all(Y, method, nsim, n_min, step_size)
            within_chain_vals.stat[j, k] = result.stat[end]
            within_chain_vals.df[j, k] = result.df[end]
            within_chain_vals.pvalue[j, k] = result.pvalue[end]
        end
    end

    return between_chain_vals, within_chain_vals, plot_vals_stat, plot_vals_pval
end

"""
    discretediag(chains::AbstractArray{<:Real,3}; frac=0.3, method=:weiss, nsim=1_000)

Compute discrete diagnostic where `method` can be one of `:weiss`, `:hangartner`,
`:DARBOOT`, `:MCBOOT`, `:billinsgley`, and `:billingsleyBOOT`.
"""
function discretediag(
    chains::AbstractArray{<:Real,3}; frac::Real=0.3, method::Symbol=:weiss, nsim::Int=1000
)
    valid_methods = (:weiss, :hangartner, :DARBOOT, :MCBOOT, :billingsley, :billingsleyBOOT)
    method in valid_methods || throw(
        ArgumentError("`method` must be one of :" * join(valid_methods, ", :", " and :")),
    )
    0 < frac < 1 || throw(ArgumentError("`frac` must be in (0,1)"))

    num_iters = size(chains, 1)
    between_chain_vals, within_chain_vals, _, _ = discretediag_sub(
        chains, frac, method, nsim, num_iters, num_iters
    )

    return between_chain_vals, within_chain_vals
end
