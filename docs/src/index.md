```@meta
CurrentModule = MCMCDiagnosticTools
```

# MCMCDiagnosticTools

## Effective sample size and $\widehat{R}$

The effective sample size (ESS) and $\widehat{R}$ can be estimated with [`ess_rhat`](@ref).

```@docs
ess_rhat
ess_rhat_bulk
ess_tail
rhat_tail
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
mcse_sbm
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