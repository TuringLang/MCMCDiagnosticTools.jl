# methods
abstract type AbstractESSMethod end

"""
    ESSMethod <: AbstractESSMethod

The `ESSMethod` uses a standard algorithm for estimating the
effective sample size of MCMC chains.

It is is based on the discussion by Vehtari et al. and uses the
biased estimator of the autocovariance, as discussed by Geyer.
In contrast to Geyer, the divisor `n - 1` is used in the estimation of
the autocovariance to obtain the unbiased estimator of the variance for lag 0.

# References

Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. Statistical Science, 473-483.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for assessing convergence of MCMC. Bayesian Analysis.
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

It is is based on the discussion by Vehtari et al. and uses the
variogram estimator of the autocorrelation function discussed by Gelman et al.

# References

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis. CRC press.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for assessing convergence of MCMC. Bayesian Analysis.
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

    # normalize autocovariance estimators by `niter - 1` instead
    # of `niter - k` to obtain
    # - unbiased estimators of the variance for lag 0
    # - biased but more stable estimators for all other lags as discussed by
    #   Geyer (1992)
    return s / (niter - 1)
end

function mean_autocov(k::Int, cache::FFTESSCache)
    # check arguments
    niter, nchains = size(cache.samples)
    0 ≤ k < niter || throw(ArgumentError("only lags ≥ 0 and < $niter are supported"))

    # compute mean autocovariance
    # we use biased but more stable estimators as discussed by Geyer (1992)
    samples_cache = cache.samples_cache
    chain_var = cache.chain_var
    return Statistics.mean(1:nchains) do i
        @inbounds(real(samples_cache[k + 1, i]) / real(samples_cache[1, i])) * chain_var[i]
    end
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

Estimate the effective sample size and the potential scale reduction of the `samples` of
shape `(draws, chains, parameters)` with the `method` and a maximum lag of `maxlag`.

By default, the computed ESS and ``\\hat{R}`` values correspond to the estimator `mean`.
Other estimators can be specified by passing a function `estimator` (see below).

`split_chains` indicates the number of chains each chain is split into.
When `split_chains > 1`, then the diagnostics check for within-chain convergence.

See also: [`ESSMethod`](@ref), [`FFTESSMethod`](@ref), [`BDAESSMethod`](@ref)

## Estimators

The ESS and ``\\hat{R}`` values can be computed for the following estimators:
- `Statistics.mean`
- `Statistics.median`
- `Statistics.std`
- `StatsBase.mad`
- `Base.Fix2(Statistics.quantile, p::Real)`
- `Base.Fix2(StatsBase.percentile, p::Real)`
"""
function ess_rhat(samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...)
    return ess_rhat(Statistics.mean, samples; kwargs...)
end
function ess_rhat(f, samples::AbstractArray{<:Union{Missing,Real},3}; kwargs...)
    x = expectand_proxy(f, samples)
    x === nothing && @error "The estimator $f is not yet supported by `ess_rhat`."
    values = _ess_rhat_mean(x; kwargs...)
    return values
end
function _ess_rhat_mean(
    chains_raw::AbstractArray{<:Union{Missing,Real},3};
    method::AbstractESSMethod=ESSMethod(),
    split_chains::Int=2,
    maxlag::Int=250,
)
    # maybe split chains
    chains = MCMCDiagnosticTools.split_chains(chains_raw, split_chains)

    # compute size of matrices
    niter, nchains, nparams = size(chains)
    ntotal = niter * nchains

    # do not compute estimates if there is only one sample or lag
    maxlag = min(maxlag, niter - 1)
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

    # for each parameter
    for (i, chains_slice) in enumerate(eachslice(chains; dims=3))
        # check that no values are missing
        if any(x -> x === missing, chains_slice)
            rhat[i] = missing
            ess[i] = missing
            continue
        end

        # split chains
        copyto!(samples, vec(chains_slice))

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
        var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected=true)
        inv_var₊ = inv(var₊)

        # estimate the potential scale reduction
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
        while k < maxlag
            # compute subsequent autocorrelation of all chains
            # by combining estimates of each chain
            ρ_even = 1 - inv_var₊ * (W - mean_autocov(k, esscache))
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

        # estimate the effective sample size
        τ = 2 * sum_pₜ - 1
        ess[i] = ntotal / τ
    end

    return ess, rhat
end

function ess_rhat_bulk(x::AbstractArray{<:Union{Missing,Real},3}; kwargs...)
    return ess_rhat(Statistics.mean, rank_normalize(x); kwargs...)
end

function ess_tail(
    x::AbstractArray{<:Union{Missing,Real},3}; tail_prob::Real=1//10, kwargs...
)
    return min.(
        ess_rhat(Base.Fix2(Statistics.quantile, tail_prob / 2), x; kwargs...)[1],
        ess_rhat(Base.Fix2(Statistics.quantile, 1 - tail_prob / 2), x; kwargs...)[1],
    )
end

rhat_tail(x; kwargs...) = ess_rhat(Statistics.mean, rank_normalize(fold(x)); kwargs...)[2]
