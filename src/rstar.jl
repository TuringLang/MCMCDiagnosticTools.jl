"""
    rstar(
        rng::Random.AbstractRNG=Random.default_rng(),
        classifier::MLJModelInterface.Supervised,
        samples,
        chain_indices::AbstractVector{Int};
        subset::Real=0.7,
        nsplit::Int=2,
        verbosity::Int=0,
    )

Compute the ``R^*`` convergence statistic of the table `samples` with the `classifier`.

`samples` must be either an `AbstractMatrix`, an `AbstractVector`, or a table
(i.e. implements the Tables.jl interface) whose rows are draws and whose columns are
parameters.

`chain_indices` indicates the chain ids of each row of `samples`.

This method supports ragged chains, i.e. chains of nonequal lengths.
"""
function rstar(
    rng::Random.AbstractRNG,
    classifier::MLJModelInterface.Supervised,
    x,
    y::AbstractVector{Int};
    subset::Real=0.7,
    nsplit::Int=2,
    verbosity::Int=0,
)
    # checks
    MLJModelInterface.nrows(x) != length(y) && throw(DimensionMismatch())
    0 < subset < 1 || throw(ArgumentError("`subset` must be a number in (0, 1)"))

    ysplit = split_chain_indices(y, nsplit)

    # randomly sub-select training and testing set
    train_ids, test_ids = shuffle_split_stratified(rng, ysplit, subset)
    0 < length(train_ids) < length(y) ||
        throw(ArgumentError("training and test data subsets must not be empty"))

    xtable = _astable(x)

    # train classifier on training data
    ycategorical = MLJModelInterface.categorical(ysplit)
    xtrain = MLJModelInterface.selectrows(xtable, train_ids)
    fitresult, _ = MLJModelInterface.fit(
        classifier, verbosity, xtrain, ycategorical[train_ids]
    )

    # compute predictions on test data
    xtest = MLJModelInterface.selectrows(xtable, test_ids)
    predictions = _predict(classifier, fitresult, xtest)

    # compute statistic
    ytest = ycategorical[test_ids]
    result = _rstar(predictions, ytest)

    return result
end

_astable(x::AbstractVecOrMat) = Tables.table(x)
_astable(x) = Tables.istable(x) ? x : throw(ArgumentError("Argument is not a valid table"))

# Workaround for https://github.com/JuliaAI/MLJBase.jl/issues/863
# `MLJModelInterface.predict` sometimes returns predictions and sometimes predictions + additional information
# TODO: Remove once the upstream issue is fixed
function _predict(model::MLJModelInterface.Model, fitresult, x)
    y = MLJModelInterface.predict(model, fitresult, x)
    return if :predict in MLJModelInterface.reporting_operations(model)
        first(y)
    else
        y
    end
end

"""
    rstar(
        rng::Random.AbstractRNG=Random.default_rng(),
        classifier::MLJModelInterface.Supervised,
        samples::AbstractArray{<:Real,3};
        subset::Real=0.7,
        nsplit::Int=2,
        verbosity::Int=0,
    )

Compute the ``R^*`` convergence statistic of the `samples` with the `classifier`.

`samples` is an array of draws with the shape `(draws, chains, parameters)`.`

This implementation is an adaption of algorithms 1 and 2 described by Lambert and Vehtari.

The `classifier` has to be a supervised classifier of the MLJ framework (see the
[MLJ documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/#model_list)
for a list of supported models). It is trained with a `subset` of the samples from each
chain. Each chain is split into `nsplit` separate chains to additionally check for
within-chain convergence. The training of the classifier can be inspected by adjusting the
`verbosity` level.

If the classifier is deterministic, i.e., if it predicts a class, the value of the ``R^*``
statistic is returned (algorithm 1). If the classifier is probabilistic, i.e., if it outputs
probabilities of classes, the scaled Poisson-binomial distribution of the ``R^*`` statistic
is returned (algorithm 2).

!!! note
    The correctness of the statistic depends on the convergence of the `classifier` used
    internally in the statistic.

# Examples

```jldoctest rstar; setup = :(using Random; Random.seed!(101))
julia> using MLJBase, MLJXGBoostInterface, Statistics

julia> samples = fill(4.0, 100, 3, 2);
```

One can compute the distribution of the ``R^*`` statistic (algorithm 2) with the
probabilistic classifier.

```jldoctest rstar
julia> distribution = rstar(XGBoostClassifier(), samples);

julia> isapprox(mean(distribution), 1; atol=0.1)
true
```

For deterministic classifiers, a single ``R^*`` statistic (algorithm 1) is returned.
Deterministic classifiers can also be derived from probabilistic classifiers by e.g.
predicting the mode. In MLJ this corresponds to a pipeline of models.

```jldoctest rstar
julia> xgboost_deterministic = Pipeline(XGBoostClassifier(); operation=predict_mode);

julia> value = rstar(xgboost_deterministic, samples);

julia> isapprox(value, 1; atol=0.2)
true
```

# References

Lambert, B., & Vehtari, A. (2020). ``R^*``: A robust MCMC convergence diagnostic with uncertainty using decision tree classifiers.
"""
function rstar(
    rng::Random.AbstractRNG,
    classifier::MLJModelInterface.Supervised,
    x::AbstractArray{<:Any,3};
    kwargs...,
)
    samples = reshape(x, :, size(x, 3))
    chain_inds = repeat(axes(x, 2); inner=size(x, 1))
    return rstar(rng, classifier, samples, chain_inds; kwargs...)
end

function rstar(classif::MLJModelInterface.Supervised, x, y::AbstractVector{Int}; kwargs...)
    return rstar(Random.default_rng(), classif, x, y; kwargs...)
end

function rstar(classif::MLJModelInterface.Supervised, x::AbstractArray{<:Any,3}; kwargs...)
    return rstar(Random.default_rng(), classif, x; kwargs...)
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
