"""
    rstar(
        rng=Random.GLOBAL_RNG,
        classifier,
        samples::AbstractMatrix,
        chain_indices::AbstractVector{Int};
        subset::Real=0.8,
        verbosity::Int=0,
    )

Compute the ``R^*`` convergence statistic of the `samples` with shape (draws, parameters)
and corresponding chains `chain_indices` with the `classifier`.

This implementation is an adaption of algorithms 1 and 2 described by Lambert and Vehtari.

The `classifier` has to be a supervised classifier of the MLJ framework (see the
[MLJ documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/#model_list)
for a list of supported models). It is trained with a `subset` of the samples. The training
of the classifier can be inspected by adjusting the `verbosity` level.

If the classifier is deterministic, i.e., if it predicts a class, the value of the ``R^*``
statistic is returned (algorithm 1). If the classifier is probabilistic, i.e., if it outputs
probabilities of classes, the scaled Poisson-binomial distribution of the ``R^*`` statistic
is returned (algorithm 2).

!!! note
    The correctness of the statistic depends on the convergence of the `classifier` used
    internally in the statistic.

# Examples

```jldoctest rstar; setup = :(using Random; Random.seed!(100))
julia> using MLJBase, MLJXGBoostInterface, Statistics

julia> samples = fill(4.0, 300, 2);

julia> chain_indices = repeat(1:3; outer=100);
```

One can compute the distribution of the ``R^*`` statistic (algorithm 2) with the
probabilistic classifier.

```jldoctest rstar
julia> distribution = rstar(XGBoostClassifier(), samples, chain_indices);

julia> isapprox(mean(distribution), 1; atol=0.1)
true
```

For deterministic classifiers, a single ``R^*`` statistic (algorithm 1) is returned.
Deterministic classifiers can also be derived from probabilistic classifiers by e.g.
predicting the mode. In MLJ this corresponds to a pipeline of models.

```jldoctest rstar
julia> xgboost_deterministic = Pipeline(XGBoostClassifier(); operation=predict_mode);

julia> value = rstar(xgboost_deterministic, samples, chain_indices);

julia> isapprox(value, 1; atol=0.2)
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
    subset::Real=0.8,
    verbosity::Int=0,
)
    # checks
    size(x, 1) != length(y) && throw(DimensionMismatch())
    0 < subset < 1 || throw(ArgumentError("`subset` must be a number in (0, 1)"))

    # randomly sub-select training and testing set
    N = length(y)
    Ntrain = round(Int, N * subset)
    0 < Ntrain < N ||
        throw(ArgumentError("training and test data subsets must not be empty"))
    ids = Random.randperm(rng, N)
    train_ids = view(ids, 1:Ntrain)
    test_ids = view(ids, (Ntrain + 1):N)

    # train classifier on training data
    ycategorical = MLJModelInterface.categorical(y)
    fitresult, _ = MLJModelInterface.fit(
        classifier, verbosity, Tables.table(x[train_ids, :]), ycategorical[train_ids]
    )

    # compute predictions on test data
    xtest = Tables.table(x[test_ids, :])
    predictions = MLJModelInterface.predict(classifier, fitresult, xtest)

    # compute statistic
    ytest = ycategorical[test_ids]
    result = _rstar(predictions, ytest)

    return result
end

function rstar(
    classif::MLJModelInterface.Supervised,
    x::AbstractMatrix,
    y::AbstractVector{Int};
    kwargs...,
)
    return rstar(Random.GLOBAL_RNG, classif, x, y; kwargs...)
end

# R⋆ for deterministic predictions (algorithm 1)
function _rstar(predictions::AbstractVector{T}, ytest::AbstractVector{T}) where {T}
    length(predictions) == length(ytest) ||
        error("numbers of predictions and targets must be equal")
    mean_accuracy = Statistics.mean(p == y for (p, y) in zip(predictions, ytest))
    nclasses = length(MLJModelInterface.classes(ytest))
    return nclasses * mean_accuracy
end

# R⋆ for probabilistic predictions (algorithm 2)
function _rstar(predictions::AbstractVector, ytest::AbstractVector)
    length(predictions) == length(ytest) ||
        error("numbers of predictions and targets must be equal")

    # create Poisson binomial distribution with support `0:length(predictions)`
    distribution = Distributions.PoissonBinomial(map(Distributions.pdf, predictions, ytest))

    # scale distribution to support in `[0, nclasses]`
    nclasses = length(MLJModelInterface.classes(ytest))
    scaled_distribution = (nclasses//length(predictions)) * distribution

    return scaled_distribution
end
