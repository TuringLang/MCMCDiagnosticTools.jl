```@meta
CurrentModule = MCMCDiagnosticTools
```

# MCMCDiagnosticTools

MCMCDiagnosticTools provides functionality for diagnosing samples generated using Markov Chain Monte Carlo.

## Background

Some methods were originally part of [Mamba.jl](https://github.com/brian-j-smith/Mamba.jl) and then [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl).
This package is a joint collaboration between the [Turing](https://turinglang.org/) and [ArviZ](https://www.arviz.org/) projects.

## Effective sample size and $\widehat{R}$

```@docs
ess
rhat
ess_rhat
```

The following `autocov_method`s are supported:

```@docs
AutocovMethod
FFTAutocovMethod
BDAAutocovMethod
```

## Monte Carlo standard error

```@docs
mcse
```

## R⋆ diagnostic

```@docs
rstar
```

## Bayesian fraction of missing information

```@docs
bfmi
```

## Other diagnostics

!!! note
    These diagnostics are older and less widely used.

```@docs
discretediag
gelmandiag
gelmandiag_multivariate
gewekediag
heideldiag
rafterydiag
```