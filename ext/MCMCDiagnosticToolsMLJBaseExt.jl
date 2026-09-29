module MCMCDiagnosticToolsMLJBaseExt

using Distributions: Distributions
using MCMCDiagnosticTools:
    MCMCDiagnosticTools, shuffle_split_stratified, split_chain_indices
using MLJModelInterface: MLJModelInterface as MMI
using Random: Random
using Statistics: Statistics
using Tables: Tables

import MCMCDiagnosticTools:
    _check_model_supports_continuous_inputs,
    _check_model_supports_multiclass_predictions,
    _check_model_supports_multiclass_targets,
    _rstar

function _rstar(
    rng::Random.AbstractRNG,
    classifier,
    x,
    y::AbstractVector{Int};
    subset::Real,
    split_chains::Int,
    verbosity::Int,
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
    return throw(
        ArgumentError(
            "unsupported types of predictions ($(typeof(predictions))) and targets ($(typeof(targets)))",
        ),
    )
end

end
