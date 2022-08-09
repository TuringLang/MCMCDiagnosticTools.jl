"""
    bfmi(energy; dims=1)

Calculate the estimated Bayesian fraction of missing information (BFMI).

When sampling with Hamiltonian Monte Carlo (HMC), BFMI quantifies how well momentum
resampling matches the marginal energy distribution.

The current advice is that values smaller than 0.3 indicate poor sampling.
However, this threshold is provisional and may change.
A BFMI value below the threshold often indicates poor adaptation of sampling parameters or
that the target distribution has heavy tails that were not well explored by the Markov
chain.

For more information, see Section 6.1 of [^Betancourt2018] or [^Betancourt2016] for a
complete account.

`energy` is either a vector of Hamiltonian energies of draws or a matrix of energies of
draws for multiple chains.
`dims` indicates the dimension in `energy` that contains the draws.
The default `dims=1` assumes `energy` has the shape `draws` or `(draws, chains)`.
If a different shape is provided, `dims` must be set accordingly.

If `energy` is a vector, a single BFMI value is returned.
Otherwise, a vector of BFMI values for each chain is returned.

# References

[^Betancourt2018]: Betancourt M. (2018).
    A Conceptual Introduction to Hamiltonian Monte Carlo.
    [arXiv:1701.02434v2](https://arxiv.org/pdf/1701.02434v2.pdf) [stat.ME]
[^Betancourt2016]: Betancourt M. (2016).
    Diagnosing Suboptimal Cotangent Disintegrations in Hamiltonian Monte Carlo.
    [arXiv:1604.00695v1](https://arxiv.org/pdf/1604.00695v1.pdf) [stat.ME]
"""
function bfmi(energy; dims=1)
    result = dropdims(mean(abs2, diff(energy; dims); dims) ./ var(energy; dims); dims)
    return iszero(ndims(result)) ? result[] : result
end
