module MCMCDiagnosticToolsDistributionsExt

using MCMCDiagnosticTools: MCMCDiagnosticTools
using Distributions: Distributions

function MCMCDiagnosticTools._rstar_distribution(predictions, ytest, nclasses::Int)
    # create Poisson binomial distribution with support `0:length(predictions)`
    distribution = Distributions.PoissonBinomial(map(Distributions.pdf, predictions, ytest))

    # scale distribution to support in `[0, nclasses]`
    scaled_distribution = (nclasses//length(predictions)) * distribution

    return scaled_distribution
end

end
