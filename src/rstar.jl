"""
    rstar(
        rng::Random.AbstractRNG=Random.default_rng(),
        classifier,
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
    classifier,
    x,
    y::AbstractVector{Int};
    subset::Real=0.7,
    split_chains::Int=2,
    verbosity::Int=0,
)
    # check the arguments
    _check_model_supports_continuous_inputs(classifier)
    _check_model_supports_multiclass_targets(classifier)
    _check_model_supports_multiclass_predictions(classifier)
    MMI.nrows(x) != length(y) && throw(DimensionMismatch())
    0 < subset < 1 || throw(ArgumentError("`subset` must be a number in (0, 1)"))

    # randomly sub-select training and testing set
    ysplit = split_chain_indices(y, split_chains)
    train_ids, test_ids = shuffle_split_stratified(rng, ysplit, subset)
    0 < length(train_ids) < length(y) ||
        throw(ArgumentError("training and test data subsets must not be empty"))

    xtable = _astable(x)
    ycategorical = MMI.categorical(ysplit)

    # train classifier on training data
    data = MMI.reformat(classifier, xtable, ycategorical)
    train_data = MMI.selectrows(classifier, train_ids, data...)
    fitresult, _ = MMI.fit(classifier, verbosity, train_data...)

    # compute predictions on test data
    # we exploit that MLJ demands that
    # reformat(model, args...)[1] = reformat(model, args[1])
    # (https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Implementing-a-data-front-end)
    test_data = MMI.selectrows(classifier, test_ids, data[1])
    predictions = _predict(classifier, fitresult, test_data...)

    # compute statistic
    ytest = ycategorical[test_ids]
    result = _rstar(MMI.scitype(predictions), predictions, ytest)

    return result
end

# check that the model supports the inputs and targets, and has predictions of the desired form
function _check_model_supports_continuous_inputs(classifier)
    # ideally we would not allow MMI.Unknown but some models do not implement the traits
    input_scitype_classifier = MMI.input_scitype(classifier)
    if input_scitype_classifier !== MMI.Unknown &&
        !(MMI.Table(MMI.Continuous) <: input_scitype_classifier)
        throw(
            ArgumentError(
                "classifier does not support tables of continuous values as inputs"
            ),
        )
    end
    return nothing
end
function _check_model_supports_multiclass_targets(classifier)
    target_scitype_classifier = MMI.target_scitype(classifier)
    if target_scitype_classifier !== MMI.Unknown &&
        !(AbstractVector{<:MMI.Finite} <: target_scitype_classifier)
        throw(
            ArgumentError(
                "classifier does not support vectors of multi-class labels as targets"
            ),
        )
    end
    return nothing
end
function _check_model_supports_multiclass_predictions(classifier)
    if !(
        MMI.predict_scitype(classifier) <: Union{
            MMI.Unknown,
            AbstractVector{<:MMI.Finite},
            AbstractVector{<:MMI.Density{<:MMI.Finite}},
        }
    )
        throw(
            ArgumentError(
                "classifier does not support vectors of multi-class labels or their densities as predictions",
            ),
        )
    end
    return nothing
end

_astable(x::AbstractVecOrMat) = Tables.table(x)
_astable(x) = Tables.istable(x) ? x : throw(ArgumentError("Argument is not a valid table"))

# Workaround for https://github.com/JuliaAI/MLJBase.jl/issues/863
# `MLJModelInterface.predict` sometimes returns predictions and sometimes predictions + additional information
# TODO: Remove once the upstream issue is fixed
function _predict(model::MMI.Model, fitresult, x)
    y = MMI.predict(model, fitresult, x)
    return if :predict in MMI.reporting_operations(model)
        first(y)
    else
        y
    end
end

"""
    rstar(
        rng::Random.AbstractRNG=Random.default_rng(),
        classifier,
        samples::AbstractArray{<:Real};
        subset::Real=0.7,
        split_chains::Int=2,
        verbosity::Int=0,
    )

Compute the ``R^*`` convergence statistic of the `samples` with the `classifier`.

`samples` is an array of draws with the shape `(draws, [chains[, parameters...]])`.`

This implementation is an adaption of algorithms 1 and 2 described by Lambert and Vehtari.

The `classifier` has to be a supervised classifier of the MLJ framework (see the
[MLJ documentation](@extref MLJ list_of_supported_models)
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
julia> using MLJBase, MLJIteration, EvoTrees, Statistics, StatisticalMeasures

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
With the MLJ framework, this can be achieved in the following way (see the
[MLJ documentation](@extref MLJ Controlling-Iterative-Models) for additional explanations):

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
predicting the mode. In MLJ this corresponds to a [pipeline](@extref MLJ Pipeline_MLJBase)
of models.

```jldoctest rstar
julia> evotree_deterministic = Pipeline(model; operation=predict_mode);

julia> value = rstar(evotree_deterministic, samples);

julia> round(value; digits=2)
1.0
```

# References

Lambert, B., & Vehtari, A. (2020). ``R^*``: A robust MCMC convergence diagnostic with uncertainty using decision tree classifiers.
"""
function rstar(rng::Random.AbstractRNG, classifier, x::AbstractArray; kwargs...)
    samples = reshape(x, size(x, 1) * size(x, 2), :)
    chain_inds = repeat(axes(x, 2); inner=size(x, 1))
    return rstar(rng, classifier, samples, chain_inds; kwargs...)
end

function rstar(classifier, x, y::AbstractVector{Int}; kwargs...)
    return rstar(Random.default_rng(), classifier, x, y; kwargs...)
end
# Fix method ambiguity issue
function rstar(rng::Random.AbstractRNG, classifier, x::AbstractVector{Int}; kwargs...)
    samples = reshape(x, length(x), :)
    chain_inds = ones(Int, length(x))
    return rstar(rng, classifier, samples, chain_inds; kwargs...)
end

function rstar(classifier, x::AbstractArray; kwargs...)
    return rstar(Random.default_rng(), classifier, x; kwargs...)
end

# R⋆ for deterministic predictions (algorithm 1)
function _rstar(
    ::Type{<:AbstractVector{<:MMI.Finite}},
    predictions::AbstractVector,
    ytest::AbstractVector,
)
    length(predictions) == length(ytest) ||
        error("numbers of predictions and targets must be equal")
    mean_accuracy = Statistics.mean(p == y for (p, y) in zip(predictions, ytest))
    nclasses = length(MMI.classes(ytest))
    return nclasses * mean_accuracy
end

# R⋆ for probabilistic predictions (algorithm 2)
function _rstar(
    ::Type{<:AbstractVector{<:MMI.Density{<:MMI.Finite}}},
    predictions::AbstractVector,
    ytest::AbstractVector,
)
    length(predictions) == length(ytest) ||
        error("numbers of predictions and targets must be equal")

    # create Poisson binomial distribution with support `0:length(predictions)`
    distribution = Distributions.PoissonBinomial(map(Distributions.pdf, predictions, ytest))

    # scale distribution to support in `[0, nclasses]`
    nclasses = length(MMI.classes(ytest))
    scaled_distribution = (nclasses//length(predictions)) * distribution

    return scaled_distribution
end

# unsupported types of predictions and targets
function _rstar(::Any, predictions, targets)
    throw(
        ArgumentError(
            "unsupported types of predictions ($(typeof(predictions))) and targets ($(typeof(targets)))",
        ),
    )
end
