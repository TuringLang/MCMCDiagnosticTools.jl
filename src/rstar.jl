"""
    rstar(
        rng=Random.GLOBAL_RNG,
        classifier::Supervised,
        samples::AbstractMatrix,
        chain_indices::AbstractVector{Int};
        iterations=classifier isa Probabilistic ? 10 : 1,
        subset=0.8,
        verbosity=0,
    )

Compute the ``R^*`` convergence diagnostic of MCMC of the `samples` with shape
(draws, parameters) and corresponding chains `chain_indices` with the `classifier`.

This implementation is an adaption of algorithms 1 and 2 described by Lambert and Vehtari.
The classifier is trained with a `subset` of the samples. The statistic is estimated with
`iterations` number of iterations. If the classifier is not probabilistic, i.e. does not
return class probabilities, it is advisable to use `iterations=1`. The training of the
classifier can be inspected by adjusting the `verbosity` level.

!!! note
    The correctness of the statistic depends on the convergence of the `classifier` used
    internally in the statistic.

# Examples

```jldoctest rstar
julia> using MLJModels, Statistics

julia> XGBoost = @load XGBoostClassifier verbosity=0;

julia> samples = fill(4.0, 300, 2);

julia> chain_indices = repeat(1:3; outer=100);

julia> stats = rstar(XGBoost(), samples, chain_indices; iterations=20);

julia> isapprox(mean(stats), 1; atol=0.1)
true
```

# References

Lambert, B., & Vehtari, A. (2020). ``R^*``: A robust MCMC convergence diagnostic with uncertainty using decision tree classifiers.
"""
function rstar(
    rng::Random.AbstractRNG,
    classifier::MLJModelInterface.Supervised,
    x::AbstractMatrix,
    y::AbstractVector{Int};
    iterations=classifier isa MLJModelInterface.Probabilistic ? 10 : 1,
    subset=0.8,
    verbosity=0,
)
    size(x, 1) != length(y) && throw(DimensionMismatch())
    iterations > 0 || throw(ArgumentError("Number of iterations has to be positive!"))

    if iterations > 1 && classifier isa MLJModelInterface.Deterministic
        @warn(
            "Classifier is not a probabilistic classifier but number of iterations is > 1."
        )
    elseif iterations == 1 && classifier isa MLJModelInterface.Probabilistic
        @warn("Classifier is probabilistic but number of iterations is equal to one.")
    end

    N = length(y)
    K = length(unique(y))

    # randomly sub-select training and testing set
    Ntrain = round(Int, N * subset)
    Ntest = N - Ntrain

    ids = Random.randperm(rng, N)
    train_ids = view(ids, 1:Ntrain)
    test_ids = view(ids, (Ntrain + 1):N)

    # train classifier using XGBoost
    fitresult, _ = MLJModelInterface.fit(
        classifier,
        verbosity,
        Tables.table(x[train_ids, :]),
        MLJModelInterface.categorical(y[train_ids]),
    )

    xtest = Tables.table(x[test_ids, :])
    ytest = view(y, test_ids)

    Rstats = map(1:iterations) do i
        return K * rstar_score(rng, classifier, fitresult, xtest, ytest)
    end
    return Rstats
end

function rstar(
    classif::MLJModelInterface.Supervised,
    x::AbstractMatrix,
    y::AbstractVector{Int};
    kwargs...,
)
    return rstar(Random.GLOBAL_RNG, classif, x, y; kwargs...)
end

function rstar_score(
    rng::Random.AbstractRNG,
    classif::MLJModelInterface.Probabilistic,
    fitresult,
    xtest,
    ytest,
)
    pred =
        DataAPI.unwrap.(
            rand.(Ref(rng), MLJModelInterface.predict(classif, fitresult, xtest))
        )
    return Statistics.mean(((p, y),) -> p == y, zip(pred, ytest))
end

function rstar_score(
    rng::Random.AbstractRNG,
    classif::MLJModelInterface.Deterministic,
    fitresult,
    xtest,
    ytest,
)
    pred = MLJModelInterface.predict(classif, fitresult, xtest)
    return Statistics.mean(((p, y),) -> p == y, zip(pred, ytest))
end
