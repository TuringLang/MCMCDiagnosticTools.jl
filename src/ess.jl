# methods
abstract type AbstractESSMethod end

"""
    ESSMethod <: AbstractESSMethod

The `ESSMethod` uses a standard algorithm for estimating the
effective sample size of MCMC chains.

It is is based on the discussion by [^VehtariGelman2021] and uses the
biased estimator of the autocovariance, as discussed by [^Geyer1992].

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
[^Geyer1992]: Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. Statistical Science, 473-483.
"""
struct ESSMethod <: AbstractESSMethod end

"""
    FFTESSMethod <: AbstractESSMethod

The `FFTESSMethod` uses a standard algorithm for estimating
the effective sample size of MCMC chains.

The algorithm is the same as the one of [`ESSMethod`](@ref) but this method uses fast
Fourier transforms (FFTs) for estimating the autocorrelation.

!!! info
    To be able to use this method, you have to load a package that implements the
    [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) interface such
    as [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) or
    [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl).
"""
struct FFTESSMethod <: AbstractESSMethod end

"""
    BDAESSMethod <: AbstractESSMethod

The `BDAESSMethod` uses a standard algorithm for estimating the effective sample size of
MCMC chains.

It is is based on the discussion by [^VehtariGelman2021]. and uses the
variogram estimator of the autocorrelation function discussed by [^BDA3].

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
[^BDA3]: Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis. CRC press.
"""
struct BDAESSMethod <: AbstractESSMethod end

# caches
struct ESSCache{T,S}
    samples::Matrix{T}
    chain_var::Vector{S}
end

struct FFTESSCache{T,S,C,P,I}
    samples::Matrix{T}
    chain_var::Vector{S}
    samples_cache::C
    plan::P
    invplan::I
end

mutable struct BDAESSCache{T,S,M}
    samples::Matrix{T}
    chain_var::Vector{S}
    mean_chain_var::M
end

function build_cache(::ESSMethod, samples::Matrix, var::Vector)
    # check arguments
    niter, nchains = size(samples)
    length(var) == nchains || throw(DimensionMismatch())

    return ESSCache(samples, var)
end

function build_cache(::FFTESSMethod, samples::Matrix, var::Vector)
    # check arguments
    niter, nchains = size(samples)
    length(var) == nchains || throw(DimensionMismatch())

    # create cache for FFT
    T = complex(eltype(samples))
    n = nextprod([2, 3], 2 * niter - 1)
    samples_cache = Matrix{T}(undef, n, nchains)

    # create plans of FFTs
    fft_plan = AbstractFFTs.plan_fft!(samples_cache, 1)
    ifft_plan = AbstractFFTs.plan_ifft!(samples_cache, 1)

    return FFTESSCache(samples, var, samples_cache, fft_plan, ifft_plan)
end

function build_cache(::BDAESSMethod, samples::Matrix, var::Vector)
    # check arguments
    nchains = size(samples, 2)
    length(var) == nchains || throw(DimensionMismatch())

    return BDAESSCache(samples, var, Statistics.mean(var))
end

update!(cache::ESSCache) = nothing

function update!(cache::FFTESSCache)
    # copy samples and add zero padding
    samples = cache.samples
    samples_cache = cache.samples_cache
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

    # compute unnormalized autocovariance
    cache.plan * samples_cache
    @. samples_cache = abs2(samples_cache)
    cache.invplan * samples_cache

    return nothing
end

function update!(cache::BDAESSCache)
    # recompute mean of within-chain variances
    cache.mean_chain_var = Statistics.mean(cache.chain_var)

    return nothing
end

function mean_autocov(k::Int, cache::ESSCache)
    # check arguments
    samples = cache.samples
    niter, nchains = size(samples)
    0 ≤ k < niter || throw(ArgumentError("only lags ≥ 0 and < $niter are supported"))

    # compute mean of unnormalized autocovariance estimates
    firstrange = 1:(niter - k)
    lastrange = (k + 1):niter
    s = Statistics.mean(1:nchains) do i
        return @inbounds LinearAlgebra.dot(
            view(samples, firstrange, i), view(samples, lastrange, i)
        )
    end

    # normalize autocovariance estimators by `niter` instead of `niter - k` to obtain biased
    # but more stable estimators for all lags as discussed by Geyer (1992)
    return s / niter
end

function mean_autocov(k::Int, cache::FFTESSCache)
    # check arguments
    niter, nchains = size(cache.samples)
    0 ≤ k < niter || throw(ArgumentError("only lags ≥ 0 and < $niter are supported"))

    # compute mean autocovariance
    # we use biased but more stable estimators as discussed by Geyer (1992)
    samples_cache = cache.samples_cache
    chain_var = cache.chain_var
    uncorrection_factor = (niter - 1)//niter  # undo corrected=true for chain_var
    result = Statistics.mean(1:nchains) do i
        @inbounds(real(samples_cache[k + 1, i]) / real(samples_cache[1, i])) * chain_var[i]
    end
    return result * uncorrection_factor
end

function mean_autocov(k::Int, cache::BDAESSCache)
    # check arguments
    samples = cache.samples
    niter, nchains = size(samples)
    0 ≤ k < niter || throw(ArgumentError("only lags ≥ 0 and < $niter are supported"))

    # compute mean autocovariance
    n = niter - k
    idxs = 1:n
    s = Statistics.mean(1:nchains) do j
        return sum(idxs) do i
            @inbounds abs2(samples[i, j] - samples[k + i, j])
        end
    end

    return cache.mean_chain_var - s / (2 * n)
end

"""
    ess_rhat(
        [estimator,]
        samples::AbstractArray{<:Union{Missing,Real},3};
        method=ESSMethod(),
        split_chains::Int=2,
        maxlag::Int=250,
    )

Estimate the effective sample size and ``\\widehat{R}`` of the `samples` of shape
`(draws, chains, parameters)` with the `method`.

`maxlag` indicates the maximum lag for which autocovariance is computed.

By default, the computed ESS and ``\\widehat{R}`` values correspond to the estimator `mean`.
Other estimators can be specified by passing a function `estimator` (see below).

`split_chains` indicates the number of chains each chain is split into.
When `split_chains > 1`, then the diagnostics check for within-chain convergence. When
`d = mod(draws, split_chains) > 0`, i.e. the chains cannot be evenly split, then 1 draw
is discarded after each of the first `d` splits within each chain.

For a given estimand, it is recommended that the ESS is at least `100 * chains` and that
``\\widehat{R} < 1.01``.[^VehtariGelman2021]

See also: [`ESSMethod`](@ref), [`FFTESSMethod`](@ref), [`BDAESSMethod`](@ref),
[`ess_rhat_bulk`](@ref), [`ess_tail`](@ref), [`rhat_tail`](@ref)

## Estimators

The ESS and ``\\widehat{R}`` values can be computed for the following estimators:
- `Statistics.mean`
- `Statistics.median`
- `Statistics.std`
- `StatsBase.mad`
- `Base.Fix2(Statistics.quantile, p::Real)`

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
"""
function ess_rhat(samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...)
    return ess_rhat(Statistics.mean, samples; kwargs...)
end
function ess_rhat(f, samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...)
    x = _expectand_proxy(f, samples)
    if x === nothing
        throw(ArgumentError("the estimator $f is not yet supported by `ess_rhat`"))
    end
    values = ess_rhat(Statistics.mean, x; kwargs...)
    return values
end
function ess_rhat(
    ::typeof(Statistics.mean),
    chains::AbstractArray{<:Union{Missing,Real},3};
    method::AbstractESSMethod=ESSMethod(),
    split_chains::Int=2,
    maxlag::Int=250,
)
    # compute size of matrices (each chain may be split!)
    niter = size(chains, 1) ÷ split_chains
    nparams = size(chains, 3)
    nchains = split_chains * size(chains, 2)
    ntotal = niter * nchains

    # discard the last pair of autocorrelations, which are poorly estimated and only matter
    # when chains have mixed poorly anyways.
    # leave the last even autocorrelation as a bias term that reduces variance for
    # case of antithetical chains, see below
    maxlag = min(maxlag, niter - 4)
    maxlag > 0 || return fill(missing, nparams), fill(missing, nparams)

    # define caches for mean and variance
    U = typeof(zero(eltype(chains)) / 1)
    T = promote_type(eltype(chains), typeof(zero(eltype(chains)) / 1))
    chain_mean = Array{T}(undef, 1, nchains)
    chain_var = Array{T}(undef, nchains)
    samples = Array{T}(undef, niter, nchains)

    # compute correction factor
    correctionfactor = (niter - 1) / niter

    # define cache for the computation of the autocorrelation
    esscache = build_cache(method, samples, chain_var)

    # define output arrays
    ess = Vector{T}(undef, nparams)
    rhat = Vector{T}(undef, nparams)

    # set maximum ess for antithetic chains, see below
    ess_max = ntotal * log10(oftype(one(T), ntotal))

    # for each parameter
    for (i, chains_slice) in enumerate(eachslice(chains; dims=3))
        # check that no values are missing
        if any(x -> x === missing, chains_slice)
            rhat[i] = missing
            ess[i] = missing
            continue
        end

        # split chains
        copyto_split!(samples, chains_slice)

        # calculate mean of chains
        Statistics.mean!(chain_mean, samples)

        # calculate within-chain variance
        @inbounds for j in 1:nchains
            chain_var[j] = Statistics.var(
                view(samples, :, j); mean=chain_mean[j], corrected=true
            )
        end
        W = Statistics.mean(chain_var)

        # compute variance estimator var₊, which accounts for between-chain variance as well
        # avoid NaN when nchains=1 and set the variance estimator var₊ to the the within-chain variance in that case
        var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected=(nchains > 1))
        inv_var₊ = inv(var₊)

        # estimate rhat
        rhat[i] = sqrt(var₊ / W)

        # center the data around 0
        samples .-= chain_mean

        # update cache
        update!(esscache)

        # compute the first two autocorrelation estimates
        # by combining autocorrelation (or rather autocovariance) estimates of each chain
        ρ_odd = 1 - inv_var₊ * (W - mean_autocov(1, esscache))
        ρ_even = one(ρ_odd) # estimate at lag 0 is known

        # sum correlation estimates
        pₜ = ρ_even + ρ_odd
        sum_pₜ = pₜ

        k = 2
        while true
            # compute subsequent autocorrelation of all chains
            # by combining estimates of each chain
            ρ_even = 1 - inv_var₊ * (W - mean_autocov(k, esscache))
            # stop summation if the next even lag would exceed maxlag. this ρ_odd is unused.
            k < maxlag - 1 || break
            ρ_odd = 1 - inv_var₊ * (W - mean_autocov(k + 1, esscache))

            # stop summation if p becomes non-positive
            Δ = ρ_even + ρ_odd
            Δ > zero(Δ) || break

            # generate a monotone sequence
            pₜ = min(Δ, pₜ)

            # update sum
            sum_pₜ += pₜ

            # update indices
            k += 2
        end

        # for antithetic chains
        # - reduce variance by averaging truncation to odd lag and truncation to next even lag
        # - prevent negative ESS for short chains by ensuring τ is nonnegative
        # See discussions in:
        # - § 3.2 of Vehtari et al. https://arxiv.org/pdf/1903.08008v5.pdf
        # - https://github.com/TuringLang/MCMCDiagnosticTools.jl/issues/40
        # - https://github.com/stan-dev/rstan/pull/618
        # - https://github.com/stan-dev/stan/pull/2774
        τ = max(0, 2 * sum_pₜ + max(0, ρ_even) - 1)

        # estimate the effective sample size
        ess[i] = min(ntotal / τ, ess_max)
    end

    return ess, rhat
end

"""
    ess_rhat_bulk(samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...)

Estimate the bulk-effective sample size and bulk-``\\widehat{R}`` values for the `samples` of
shape `(draws, chains, parameters)`.

For a description of `kwargs`, see [`ess_rhat`](@ref).

The bulk-ESS and bulk-``\\widehat{R}`` are variants of ESS and ``\\widehat{R}`` that
diagnose poor convergence in the bulk of the distribution due to trends or different
locations of the chains. While it is conceptually related to [`ess_rhat`](@ref) for
`Statistics.mean`, it is well-defined even if the chains do not have finite variance.[^VehtariGelman2021]

Bulk-ESS and bulk-``\\widehat{R}`` are computed by rank-normalizing the samples and then
computing `ess_rhat`. For each parameter, rank-normalization proceeds by first ranking the
inputs using "tied ranking" and then transforming the ranks to normal quantiles so that the
result is standard normally distributed. The transform is monotonic.

See also: [`ess_tail`](@ref), [`rhat_tail`](@ref)

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
"""
function ess_rhat_bulk(x::AbstractArray{<:Union{Missing,Real},3}; kwargs...)
    return ess_rhat(Statistics.mean, _rank_normalize(x); kwargs...)
end

"""
    ess_tail(samples::AbstractArray{<:Union{Missing,Real},3}; tail_prob=1//10, kwargs...)

Estimate the tail-effective sample size and for the `samples` of shape
`(draws, chains, parameters)`.

For a description of `kwargs`, see [`ess_rhat`](@ref).

The tail-ESS diagnoses poor convergence in the tails of the distribution. Specifically, it
is the minimum of the ESS of the estimate of the symmetric quantiles where `tail_prob` is
the probability in the tails. For example, with the default `tail_prob=1//10`, the tail-ESS
is the minimum of the ESS of the 0.5 and 0.95 sample quantiles.[^VehtariGelman2021]

See also: [`ess_rhat_bulk`](@ref), [`rhat_tail`](@ref)

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
"""
function ess_tail(
    x::AbstractArray{<:Union{Missing,Real},3}; tail_prob::Real=1//10, kwargs...
)
    # workaround for https://github.com/JuliaStats/Statistics.jl/issues/136
    T = Base.promote_eltype(x, tail_prob)
    return min.(
        ess_rhat(Base.Fix2(Statistics.quantile, T(tail_prob / 2)), x; kwargs...)[1],
        ess_rhat(Base.Fix2(Statistics.quantile, T(1 - tail_prob / 2)), x; kwargs...)[1],
    )
end

"""
    rhat_tail(samples::AbstractArray{Union{Real,Missing},3}; kwargs...)

Estimate the tail-``\\widehat{R}`` diagnostic for the `samples` of shape
`(draws, chains, parameters)`.

For a description of `kwargs`, see [`ess_rhat`](@ref).

The tail-``\\widehat{R}`` diagnostic is a variant of ``\\widehat{R}`` that diagnoses poor
convergence in the tails of the distribution. In particular, it can detect chains that have
similar locations but different scales.[^VehtariGelman2021]

For each parameter matrix of draws `x` with size `(draws, chains)`, it is calculated by
computing bulk-``\\widehat{R}`` on the absolute deviation of the draws from the median:
`abs.(x .- median(x))`.

See also: [`ess_tail`](@ref), [`ess_rhat_bulk`](@ref)

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
"""
rhat_tail(x; kwargs...) = ess_rhat_bulk(_fold_around_median(x); kwargs...)[2]

# Compute an expectand `z` such that ``\\textrm{mean-ESS}(z) ≈ \\textrm{f-ESS}(x)``.
# If no proxy expectand for `f` is known, `nothing` is returned.
_expectand_proxy(f, x) = nothing
function _expectand_proxy(::typeof(Statistics.median), x)
    return x .≤ Statistics.median(x; dims=(1, 2))
end
function _expectand_proxy(::typeof(Statistics.std), x)
    return (x .- Statistics.mean(x; dims=(1, 2))) .^ 2
end
function _expectand_proxy(::typeof(StatsBase.mad), x)
    x_folded = _fold_around_median(x)
    return _expectand_proxy(Statistics.median, x_folded)
end
function _expectand_proxy(f::Base.Fix2{typeof(Statistics.quantile),<:Real}, x)
    y = similar(x, Bool)
    # currently quantile does not support a dims keyword argument
    for (xi, yi) in zip(eachslice(x; dims=3), eachslice(y; dims=3))
        yi .= xi .≤ f(vec(xi))
    end
    return y
end
