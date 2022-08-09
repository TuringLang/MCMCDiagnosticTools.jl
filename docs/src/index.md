```@meta
CurrentModule = MCMCDiagnosticTools
```

# MCMCDiagnosticTools

## Effective sample size and potential scale reduction

The effective sample size (ESS) and the potential scale reduction can be
estimated with [`ess_rhat`](@ref).

```@docs
ess_rhat
```

The following methods are supported:

```@docs
ESSMethod
FFTESSMethod
BDAESSMethod
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