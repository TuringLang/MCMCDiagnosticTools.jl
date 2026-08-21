# methods
abstract type AbstractAutocovMethod end

const _DOC_SPLIT_CHAINS = """`split_chains` indicates the number of chains each chain is split into.
                          When `split_chains > 1`, then the diagnostics check for within-chain convergence. When
                          `d = mod(draws, split_chains) > 0`, i.e. the chains cannot be evenly split, then 1 draw
                          is discarded after each of the first `d` splits within each chain. Ragged chains are
                          split independently."""

const _DOC_RAGGED_SAMPLES = """
Alternatively, `samples` may be a vector of arrays, one per chain, where chain `i` has
shape `(draws_i, [parameters...])`. All chains must have identical parameter axes. Ragged
chains are not padded or truncated to a common length.

For ragged inputs, chain statistics are weighted equally and the variance estimate used by
ESS and ``\\widehat{R}`` is ``W + \\operatorname{var}(\\text{chain means})``. Rectangular
inputs retain the existing finite-draw correction for backwards compatibility, so equal-
length rectangular and vector-of-chains inputs can differ slightly."""

const _DOC_RHAT_KIND = """
## Kinds of ``\\widehat{R}``

The following `kind`s are supported:
- `:rank`: maximum of ``\\widehat{R}`` with `kind=:bulk` and `kind=:tail`.
- `:bulk`: basic ``\\widehat{R}`` computed on rank-normalized draws. This kind diagnoses
    poor convergence in the bulk of the distribution due to trends or different locations of
    the chains.
- `:tail`: ``\\widehat{R}`` computed on draws folded around the median and then
    rank-normalized. This kind diagnoses poor convergence in the tails of the distribution
    due to different scales of the chains.
- `:basic`: Classic ``\\widehat{R}``."""

"""
    AutocovMethod <: AbstractAutocovMethod

The `AutocovMethod` uses a standard algorithm for estimating the mean autocovariance of MCMC
chains.

It is is based on the discussion by [^VehtariGelman2021] and uses the
biased estimator of the autocovariance, as discussed by [^Geyer1992].

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
[^Geyer1992]: Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. Statistical Science, 473-483.
"""
struct AutocovMethod <: AbstractAutocovMethod end

"""
    FFTAutocovMethod <: AbstractAutocovMethod

The `FFTAutocovMethod` uses a standard algorithm for estimating the mean autocovariance of
MCMC chains.

The algorithm is the same as the one of [`AutocovMethod`](@ref) but this method uses fast
Fourier transforms (FFTs) for estimating the autocorrelation.

!!! info
    To be able to use this method, you have to load a package that implements the
    [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) interface such
    as [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) or
    [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl).
"""
struct FFTAutocovMethod <: AbstractAutocovMethod end

"""
    BDAAutocovMethod <: AbstractAutocovMethod

The `BDAAutocovMethod` uses a standard algorithm for estimating the mean autocovariance of
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
struct BDAAutocovMethod <: AbstractAutocovMethod end

# caches
struct AutocovCache{S,V}
    samples::S
    chain_var::V
end

struct FFTAutocovCache{S,V,C,P,I}
    samples::S
    chain_var::V
    samples_cache::C
    plan::P
    invplan::I
end

mutable struct BDAAutocovCache{S,V,M}
    samples::S
    chain_var::V
    mean_chain_var::M
end

function build_cache(::AutocovMethod, samples, var::Vector)
    # check arguments
    length(var) == _nchains(samples) || throw(DimensionMismatch())

    return AutocovCache(samples, var)
end

function build_cache(::FFTAutocovMethod, samples, var::Vector)
    # check arguments
    nchains = _nchains(samples)
    length(var) == nchains || throw(DimensionMismatch())

    # create cache for FFT
    T = complex(_sample_eltype(samples))
    niter = _max_draws(samples)
    n = nextprod([2, 3], 2 * niter - 1)
    samples_cache = Matrix{T}(undef, n, nchains)

    # create plans of FFTs
    fft_plan = AbstractFFTs.plan_fft!(samples_cache, 1)
    ifft_plan = AbstractFFTs.plan_ifft!(samples_cache, 1)

    return FFTAutocovCache(samples, var, samples_cache, fft_plan, ifft_plan)
end

function build_cache(::BDAAutocovMethod, samples, var::Vector)
    # check arguments
    length(var) == _nchains(samples) || throw(DimensionMismatch())

    return BDAAutocovCache(samples, var, Statistics.mean(var))
end

update!(cache::AutocovCache) = nothing

function update!(cache::FFTAutocovCache)
    # copy samples and add zero padding
    samples = cache.samples
    samples_cache = cache.samples_cache
    _copyto_fft!(samples_cache, samples)

    # compute unnormalized autocovariance
    cache.plan * samples_cache
    @. samples_cache = abs2(samples_cache)
    cache.invplan * samples_cache

    return nothing
end

function update!(cache::BDAAutocovCache)
    # recompute mean of within-chain variances
    cache.mean_chain_var = Statistics.mean(cache.chain_var)

    return nothing
end

function mean_autocov(k::Int, cache::AutocovCache)
    # check arguments
    samples = cache.samples
    niter = _min_draws(samples)
    0 ≤ k < niter || throw(ArgumentError("only lags ≥ 0 and < $niter are supported"))

    return _mean_autocov(k, samples)
end

function _mean_autocov(k, samples::AbstractMatrix)
    niter, nchains = size(samples)
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
function _mean_autocov(k, samples::AbstractVector{<:AbstractVector})
    return Statistics.mean(samples) do chain
        niter = length(chain)
        return @inbounds LinearAlgebra.dot(
            view(chain, 1:(niter - k)), view(chain, (k + 1):niter)
        ) / niter
    end
end

function mean_autocov(k::Int, cache::FFTAutocovCache)
    # check arguments
    niter = _min_draws(cache.samples)
    0 ≤ k < niter || throw(ArgumentError("only lags ≥ 0 and < $niter are supported"))

    return _mean_autocov(k, cache.samples_cache, cache.samples, cache.chain_var)
end

function _mean_autocov(k, samples_cache, samples::AbstractMatrix, chain_var)
    niter, nchains = size(samples)
    # compute mean autocovariance
    # we use biased but more stable estimators as discussed by Geyer (1992)
    uncorrection_factor = (niter - 1)//niter  # undo corrected=true for chain_var
    result = Statistics.mean(1:nchains) do i
        return @inbounds(real(samples_cache[k + 1, i]) / real(samples_cache[1, i])) *
               chain_var[i]
    end
    return result * uncorrection_factor
end
function _mean_autocov(
    k, samples_cache, samples::AbstractVector{<:AbstractVector}, chain_var
)
    return Statistics.mean(eachindex(samples)) do i
        niter = length(samples[i])
        uncorrection_factor = (niter - 1)//niter
        return @inbounds(
            real(samples_cache[k + 1, i]) / real(samples_cache[1, i]) * chain_var[i]
        ) * uncorrection_factor
    end
end

function mean_autocov(k::Int, cache::BDAAutocovCache)
    # check arguments
    samples = cache.samples
    niter = _min_draws(samples)
    0 ≤ k < niter || throw(ArgumentError("only lags ≥ 0 and < $niter are supported"))

    return _mean_autocov(k, samples, cache.mean_chain_var)
end

function _mean_autocov(k, samples::AbstractMatrix, mean_chain_var)
    niter, nchains = size(samples)
    # compute mean autocovariance
    n = niter - k
    idxs = 1:n
    s = Statistics.mean(1:nchains) do j
        return sum(idxs) do i
            @inbounds abs2(samples[i, j] - samples[k + i, j])
        end
    end

    return mean_chain_var - s / (2 * n)
end
function _mean_autocov(k, samples::AbstractVector{<:AbstractVector}, mean_chain_var)
    s = Statistics.mean(samples) do chain
        n = length(chain) - k
        return sum(1:n) do i
            @inbounds abs2(chain[i] - chain[k + i])
        end / (2 * n)
    end
    return mean_chain_var - s
end

"""
    ess(
        samples::AbstractArray{<:Union{Missing,Real}};
        kind=:bulk,
        relative::Bool=false,
        autocov_method=AutocovMethod(),
        split_chains::Int=2,
        maxlag::Int=250,
        kwargs...
    )
    ess(samples::AbstractVector{<:AbstractArray{<:Union{Missing,Real}}}; kwargs...)

Estimate the effective sample size (ESS) of the `samples` of shape
`(draws, [chains[, parameters...]])` with the `autocov_method`.

$_DOC_RAGGED_SAMPLES

Optionally, the `kind` of ESS estimate to be computed can be specified (see below). Some
`kind`s accept additional `kwargs`.

If `relative` is `true`, the relative ESS is returned, i.e. the ESS divided by the number
of retained draws.

$_DOC_SPLIT_CHAINS There must be at least 3 draws in each chain after splitting.

`maxlag` indicates the maximum lag for which autocovariance is computed and must be greater
than 0. For ragged inputs it is capped based on the shortest chain after splitting.

For a given estimand, it is recommended that the ESS is at least `100 * chains` and that
``\\widehat{R} < 1.01``.[^VehtariGelman2021]

See also: [`AutocovMethod`](@ref), [`FFTAutocovMethod`](@ref), [`BDAAutocovMethod`](@ref),
[`rhat`](@ref), [`ess_rhat`](@ref), [`mcse`](@ref)

## Kinds of ESS estimates

If `kind` isa a `Symbol`, it may take one of the following values:
- `:bulk`: basic ESS computed on rank-normalized draws. This kind diagnoses poor convergence
    in the bulk of the distribution due to trends or different locations of the chains.
- `:tail`: minimum of the quantile-ESS for the symmetric quantiles where
    `tail_prob=0.1` is the probability in the tails. This kind diagnoses poor convergence in
    the tails of the distribution. If this kind is chosen, `kwargs` may contain a
    `tail_prob` keyword.
- `:basic`: basic ESS, equivalent to specifying `kind=Statistics.mean`.

!!! note
    While Bulk-ESS is conceptually related to basic ESS, it is well-defined even if the
    chains do not have finite variance.[^VehtariGelman2021] For each parameter,
    rank-normalization proceeds by first ranking the inputs using "tied ranking" and then
    transforming the ranks to normal quantiles so that the result is standard normally
    distributed. This transform is monotonic.

Otherwise, `kind` specifies one of the following estimators, whose ESS is to be estimated:
- [`Statistics.mean`](@extref)
- [`Statistics.median`](@extref)
- [`Statistics.std`](@extref)
- [`StatsBase.mad`](@extref)
- `Base.Fix2(Statistics.quantile, p::Real)`

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)
"""
function ess(samples::_DiagnosticSamples; kind=:bulk, kwargs...)
    _validate_samples(samples)
    # if we just call _ess(Val(kind), ...) Julia cannot infer the return type with default
    # const-propagation. We keep this type-inferrable by manually dispatching to the cases.
    if kind === :bulk
        return _ess(Val(:bulk), samples; kwargs...)
    elseif kind === :tail
        return _ess(Val(:tail), samples; kwargs...)
    elseif kind === :basic
        return _ess(Val(:basic), samples; kwargs...)
    elseif kind isa Symbol
        throw(ArgumentError("the `kind` `$kind` is not supported by `ess`"))
    else
        return _ess(kind, samples; kwargs...)
    end
end
function _ess(estimator, samples::_DiagnosticSamples; kwargs...)
    x = _expectand_proxy(estimator, samples)
    if x === nothing
        throw(ArgumentError("the estimator $estimator is not yet supported by `ess`"))
    end
    return _ess(Val(:basic), x; kwargs...)
end
function _ess(kind::Val, samples::_DiagnosticSamples; kwargs...)
    return _ess_rhat(kind, samples; kwargs...).ess
end
function _ess(::Val{:tail}, x::_DiagnosticSamples; tail_prob::Real=1//10, kwargs...)
    # workaround for https://github.com/JuliaStats/Statistics.jl/issues/136
    T = float(_promote_sample_eltype(x, tail_prob))
    pl = convert(T, tail_prob / 2)
    pu = convert(T, 1 - tail_prob / 2)
    S_lower = _ess(Base.Fix2(Statistics.quantile, pl), x; kwargs...)
    S_upper = _ess(Base.Fix2(Statistics.quantile, pu), x; kwargs...)
    return map(min, S_lower, S_upper)
end

"""
    rhat(samples::AbstractArray{Union{Real,Missing}}; kind::Symbol=:rank, split_chains=2)
    rhat(
        samples::AbstractVector{<:AbstractArray{<:Union{Missing,Real}}};
        kind::Symbol=:rank,
        split_chains=2,
    )

Compute the ``\\widehat{R}`` diagnostics for each parameter in `samples` of shape
`(draws, [chains[, parameters...]])`.[^VehtariGelman2021]

$_DOC_RAGGED_SAMPLES

`kind` indicates the kind of ``\\widehat{R}`` to compute (see extended help).

See also [`ess`](@ref), [`ess_rhat`](@ref), [`rhat_nested`](@ref), [`rstar`](@ref)

[^VehtariGelman2021]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    Rank-normalization, folding, and localization: An improved ``\\widehat {R}`` for
    assessing convergence of MCMC. Bayesian Analysis.
    doi: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
    arXiv: [1903.08008](https://arxiv.org/abs/1903.08008)

# Extended help

$_DOC_SPLIT_CHAINS

$_DOC_RHAT_KIND
"""
function rhat(samples::_DiagnosticSamples; kind::Symbol=:rank, kwargs...)
    _validate_samples(samples)
    # if we just call _rhat(Val(kind), ...) Julia cannot infer the return type with default
    # const-propagation. We keep this type-inferrable by manually dispatching to the cases.
    if kind === :rank
        return _rhat(Val(:rank), samples; kwargs...)
    elseif kind === :bulk
        return _rhat(Val(:bulk), samples; kwargs...)
    elseif kind === :tail
        return _rhat(Val(:tail), samples; kwargs...)
    elseif kind === :basic
        return _rhat(Val(:basic), samples; kwargs...)
    else
        return throw(ArgumentError("the `kind` `$kind` is not supported by `rhat`"))
    end
end
function _rhat(::Val{:basic}, chains::_DiagnosticSamples; kwargs...)
    # define output array
    T0 = _sample_eltype(chains)
    T = promote_type(T0, typeof(zero(T0) / 1))
    rhat = _similar_params(chains, T)

    if T !== Missing
        _rhat_basic!(rhat, chains; kwargs...)
    end

    return _maybescalar(rhat)
end
function _rhat_basic!(
    rhat::AbstractArray{T}, chains::_DiagnosticSamples; split_chains::Int=2
) where {T<:Union{Missing,Real}}
    # compute size of matrices (each chain may be split!)
    nchains = _nsplit_chains(chains, split_chains)

    # define caches for mean and variance
    chain_mean = Array{T}(undef, 1, nchains)
    chain_var = Array{T}(undef, nchains)
    samples = _allocate_split(chains, T, split_chains)

    # compute correction factor
    correctionfactor = _rhat_correction(chains, split_chains)

    # for each parameter
    for (i, chains_slice) in zip(eachindex(rhat), _eachparam(chains))
        # check that no values are missing
        if _any_missing(chains_slice)
            rhat[i] = missing
            continue
        end

        # split chains
        copyto_split!(samples, chains_slice)

        # calculate means and variances of chains
        _mean_var!(chain_mean, chain_var, samples)
        W = Statistics.mean(chain_var)

        # compute variance estimator var₊, which accounts for between-chain variance as well
        # avoid NaN when nchains=1 and set the variance estimator var₊ to the the within-chain variance in that case
        var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected=(nchains > 1))

        # estimate rhat
        rhat[i] = sqrt(var₊ / W)
    end
    return rhat
end
function _rhat(::Val{:bulk}, x::_DiagnosticSamples; kwargs...)
    return _rhat(Val(:basic), _rank_normalize(x); kwargs...)
end
function _rhat(::Val{:tail}, x::_DiagnosticSamples; kwargs...)
    return _rhat(Val(:bulk), _fold_around_median(x); kwargs...)
end
function _rhat(::Val{:rank}, x::_DiagnosticSamples; kwargs...)
    Rbulk = _rhat(Val(:bulk), x; kwargs...)
    Rtail = _rhat(Val(:tail), x; kwargs...)
    return map(max, Rtail, Rbulk)
end

"""
    ess_rhat(
        samples::AbstractArray{<:Union{Missing,Real}};
        kind::Symbol=:rank,
        kwargs...,
    ) -> NamedTuple{(:ess, :rhat)}
    ess_rhat(
        samples::AbstractVector{<:AbstractArray{<:Union{Missing,Real}}};
        kind::Symbol=:rank,
        kwargs...,
    ) -> NamedTuple{(:ess, :rhat)}

Estimate the effective sample size and ``\\widehat{R}`` of the `samples` of shape
`(draws, [chains[, parameters...]])`.

$_DOC_RAGGED_SAMPLES

When both ESS and ``\\widehat{R}`` are needed, this method is often more efficient than
calling `ess` and `rhat` separately.

See [`rhat`](@ref) for a description of supported `kind`s and [`ess`](@ref) for a
description of `kwargs`.
"""
function ess_rhat(samples::_DiagnosticSamples; kind::Symbol=:rank, kwargs...)
    _validate_samples(samples)
    # if we just call _ess_rhat(Val(kind), ...) Julia cannot infer the return type with
    # default const-propagation. We keep this type-inferrable by manually dispatching to the
    # cases.
    if kind === :rank
        return _ess_rhat(Val(:rank), samples; kwargs...)
    elseif kind === :bulk
        return _ess_rhat(Val(:bulk), samples; kwargs...)
    elseif kind === :tail
        return _ess_rhat(Val(:tail), samples; kwargs...)
    elseif kind === :basic
        return _ess_rhat(Val(:basic), samples; kwargs...)
    else
        return throw(ArgumentError("the `kind` `$kind` is not supported by `ess_rhat`"))
    end
end
function _ess_rhat(
    ::Val{:basic},
    chains::_DiagnosticSamples;
    split_chains::Int=2,
    maxlag::Int=250,
    kwargs...,
)
    # define output arrays
    T0 = _sample_eltype(chains)
    T = promote_type(T0, typeof(zero(T0) / 1))
    ess = _similar_params(chains, T)
    rhat = _similar_params(chains, T)

    # compute number of iterations (each chain may be split!)
    niter = _min_split_draws(chains, split_chains)

    if !(niter > 4)
        # discard the last pair of autocorrelations, which are poorly estimated and only matter
        # when chains have mixed poorly anyways.
        # leave the last even autocorrelation as a bias term that reduces variance for
        # case of antithetical chains, see below
        @warn "number of draws after splitting must be >4 but is $niter. ESS cannot be computed."
        fill!(ess, NaN)
        _rhat_basic!(rhat, chains; split_chains)
    elseif T !== Missing
        maxlag > 0 || throw(DomainError(maxlag, "maxlag must be >0."))
        maxlag = min(maxlag, niter - 4)
        _ess_rhat_basic!(ess, rhat, chains; split_chains, maxlag, kwargs...)
    end

    return (; ess=_maybescalar(ess), rhat=_maybescalar(rhat))
end
function _ess_rhat_basic!(
    ess::TA,
    rhat::TA,
    chains::_DiagnosticSamples;
    relative::Bool=false,
    autocov_method::AbstractAutocovMethod=AutocovMethod(),
    split_chains::Int=2,
    maxlag::Int=250,
) where {T<:Union{Missing,Real},TA<:AbstractArray{T}}
    # compute size of matrices (each chain may be split!)
    nchains = _nsplit_chains(chains, split_chains)
    ntotal = _total_split_draws(chains, split_chains)

    # define caches for mean and variance
    chain_mean = Array{T}(undef, 1, nchains)
    chain_var = Array{T}(undef, nchains)
    samples = _allocate_split(chains, T, split_chains)

    # compute correction factor
    correctionfactor = _rhat_correction(chains, split_chains)

    # define cache for the computation of the autocorrelation
    esscache = build_cache(autocov_method, samples, chain_var)

    # set maximum relative ess for antithetic chains, see below
    rel_ess_max = log10(oftype(one(T), ntotal))

    # for each parameter
    for (i, chains_slice) in zip(eachindex(ess), _eachparam(chains))
        # check that no values are missing
        if _any_missing(chains_slice)
            ess[i] = missing
            rhat[i] = missing
            continue
        end

        # split chains
        copyto_split!(samples, chains_slice)

        # calculate means and variances of chains
        _mean_var!(chain_mean, chain_var, samples)
        W = Statistics.mean(chain_var)

        # compute variance estimator var₊, which accounts for between-chain variance as well
        # avoid NaN when nchains=1 and set the variance estimator var₊ to the the within-chain variance in that case
        var₊ = correctionfactor * W + Statistics.var(chain_mean; corrected=(nchains > 1))
        inv_var₊ = inv(var₊)

        # estimate rhat
        rhat[i] = sqrt(var₊ / W)

        # center the data around 0
        _center!(samples, chain_mean)

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
        while k < (maxlag - 1)
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
        # for antithetic chains
        # - reduce variance by averaging truncation to odd lag and truncation to next even lag
        # - prevent negative ESS for short chains by ensuring τ is nonnegative
        # See discussions in:
        # - § 3.2 of Vehtari et al. https://arxiv.org/pdf/1903.08008v5.pdf
        # - https://github.com/TuringLang/MCMCDiagnosticTools.jl/issues/40
        # - https://github.com/stan-dev/rstan/pull/618
        # - https://github.com/stan-dev/stan/pull/2774
        ρ_even = maxlag > 1 ? 1 - inv_var₊ * (W - mean_autocov(k, esscache)) : zero(ρ_even)
        τ = max(0, 2 * sum_pₜ + max(0, ρ_even) - 1)

        # estimate the relative effective sample size
        ess[i] = min(inv(τ), rel_ess_max)
    end

    if !relative
        # absolute effective sample size
        ess .*= ntotal
    end

    return (; ess, rhat)
end
function _ess_rhat(::Val{:bulk}, x::_DiagnosticSamples; kwargs...)
    return _ess_rhat(Val(:basic), _rank_normalize(x); kwargs...)
end
function _ess_rhat(kind::Val{:tail}, x::_DiagnosticSamples; split_chains::Int=2, kwargs...)
    S = _ess(kind, x; split_chains=split_chains, kwargs...)
    R = _rhat(kind, x; split_chains=split_chains)
    return (ess=S, rhat=R)
end
function _ess_rhat(::Val{:rank}, x::_DiagnosticSamples; split_chains::Int=2, kwargs...)
    Sbulk, Rbulk = _ess_rhat(Val(:bulk), x; split_chains=split_chains, kwargs...)
    Rtail = _rhat(Val(:tail), x; split_chains=split_chains)
    Rrank = map(max, Rtail, Rbulk)
    return (ess=Sbulk, rhat=Rrank)
end

# Compute an expectand `z` such that ``\\textrm{mean-ESS}(z) ≈ \\textrm{f-ESS}(x)``.
# If no proxy expectand for `f` is known, `nothing` is returned.
_expectand_proxy(f, x) = nothing
_expectand_proxy(::typeof(Statistics.mean), x) = x
function _expectand_proxy(::typeof(Statistics.median), x)
    y = _similar_samples(x)
    # avoid using the `dims` keyword for median because it
    # - can error for Union{Missing,Real} (https://github.com/JuliaStats/Statistics.jl/issues/8)
    # - is type-unstable (https://github.com/JuliaStats/Statistics.jl/issues/39)
    for (xi, yi) in zip(_eachparam(x), _eachparam(y))
        _threshold!(yi, xi, Statistics.median(_pool(xi)))
    end
    return y
end
function _expectand_proxy(::typeof(Statistics.std), x)
    return _squared_deviations(x)
end
function _expectand_proxy(::typeof(StatsBase.mad), x)
    x_folded = _fold_around_median(x)
    return _expectand_proxy(Statistics.median, x_folded)
end
function _expectand_proxy(f::Base.Fix2{typeof(Statistics.quantile),<:Real}, x)
    y = _similar_samples(x)
    # currently quantile does not support a dims keyword argument
    for (xi, yi) in zip(_eachparam(x), _eachparam(y))
        if _any_missing(xi)
            # quantile function raises an error if there are missing values
            _fill_samples!(yi, missing)
        else
            _threshold!(yi, xi, f(_pool(xi)))
        end
    end
    return y
end
