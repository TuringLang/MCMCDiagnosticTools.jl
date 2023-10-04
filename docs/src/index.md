```@meta
CurrentModule = MCMCDiagnosticTools
```

# MCMCDiagnosticTools

MCMCDiagnosticTools provides functionality for diagnosing samples generated using Markov Chain Monte Carlo.

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

```@docs
discretediag
gelmandiag
gelmandiag_multivariate
gewekediag
heideldiag
rafterydiag
```