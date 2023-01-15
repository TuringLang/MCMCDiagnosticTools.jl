"""
    rstar(
        rng::Random.AbstractRNG=Random.default_rng(),
        classifier::MLJModelInterface.Supervised,
        samples,
        chain_indices::AbstractVector{Int};
        subset::Real=0.7,
        split_chains::Int=2,
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
    split_chains::Int=2,
    verbosity::Int=0,
)
    # checks
    MLJModelInterface.nrows(x) != length(y) && throw(DimensionMismatch())
    0 < subset < 1 || throw(ArgumentError("`subset` must be a number in (0, 1)"))

    ysplit = split_chain_indices(y, split_chains)

    # randomly sub-select training and testing set
    train_ids, test_ids = shuffle_split_stratified(rng, ysplit, subset)
    0 < length(train_ids) < length(y) ||
        throw(ArgumentError("training and test data subsets must not be empty"))

    xtable = _astable(x)
    ycategorical = MLJModelInterface.categorical(ysplit)
    xdata, ydata = MLJModelInterface.reformat(classifier, xtable, ycategorical)

    # train classifier on training data
    xtrain, ytrain = MLJModelInterface.selectrows(classifier, train_ids, xdata, ydata)
    fitresult, _ = MLJModelInterface.fit(classifier, verbosity, xtrain, ytrain)

    # compute predictions on test data
    xtest, = MLJModelInterface.selectrows(classifier, test_ids, xdata)
    ytest = ycategorical[test_ids]
    predictions = _predict(classifier, fitresult, xtest)

    # compute statistic
    result = _rstar(classifier, predictions, ytest)

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
        split_chains::Int=2,
        verbosity::Int=0,
    )

Compute the ``R^*`` convergence statistic of the `samples` with the `classifier`.

`samples` is an array of draws with the shape `(draws, chains, parameters)`.`

This implementation is an adaption of algorithms 1 and 2 described by Lambert and Vehtari.

The `classifier` has to be a supervised classifier of the MLJ framework (see the
[MLJ documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/#model_list)
for a list of supported models). It is trained with a `subset` of the samples from each
chain. Each chain is split into `split_chains` separate chains to additionally check for
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
julia> using MLJBase, MLJIteration, EvoTrees, Statistics

julia> samples = fill(4.0, 100, 3, 2);
```

One can compute the distribution of the ``R^*`` statistic (algorithm 2) with a
probabilistic classifier.
For instance, we can use a gradient-boosted trees model with `nrounds = 100` sequentially stacked trees and learning rate `eta = 0.05`:

```jldoctest rstar
julia> model = EvoTreeClassifier(; nrounds=100, eta=0.05);

julia> distribution = rstar(model, samples);

julia> round(mean(distribution); digits=2)
1.0f0
```

Note, however, that it is recommended to determine `nrounds` based on early-stopping.
With the MLJ framework, this can be achieved in the following way (see the [MLJ documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/) for additional explanations):

```jldoctest rstar
julia> model = IteratedModel(;
           model=EvoTreeClassifier(; eta=0.05),
           iteration_parameter=:nrounds,
           resampling=Holdout(),
           measures=log_loss,
           controls=[Step(5), Patience(2), NumberLimit(100)],
           retrain=true,
       );

julia> distribution = rstar(model, samples);

julia> round(mean(distribution); digits=2)
1.0f0
```

For deterministic classifiers, a single ``R^*`` statistic (algorithm 1) is returned.
Deterministic classifiers can also be derived from probabilistic classifiers by e.g.
predicting the mode. In MLJ this corresponds to a pipeline of models.

```jldoctest rstar
julia> evotree_deterministic = Pipeline(model; operation=predict_mode);

julia> value = rstar(evotree_deterministic, samples);

julia> round(value; digits=2)
1.0
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
function _rstar(
    ::MLJModelInterface.Deterministic, predictions::AbstractVector, ytest::AbstractVector
)
    length(predictions) == length(ytest) ||
        error("numbers of predictions and targets must be equal")
    mean_accuracy = Statistics.mean(p == y for (p, y) in zip(predictions, ytest))
    nclasses = length(MLJModelInterface.classes(ytest))
    return nclasses * mean_accuracy
end

# R⋆ for probabilistic predictions (algorithm 2)
function _rstar(
    ::MLJModelInterface.Probabilistic, predictions::AbstractVector, ytest::AbstractVector
)
    length(predictions) == length(ytest) ||
        error("numbers of predictions and targets must be equal")

    # create Poisson binomial distribution with support `0:length(predictions)`
    distribution = Distributions.PoissonBinomial(map(Distributions.pdf, predictions, ytest))

    # scale distribution to support in `[0, nclasses]`
    nclasses = length(MLJModelInterface.classes(ytest))
    scaled_distribution = (nclasses//length(predictions)) * distribution

    return scaled_distribution
end
