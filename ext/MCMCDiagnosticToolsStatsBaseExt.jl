module MCMCDiagnosticToolsStatsBaseExt

using MCMCDiagnosticTools: MCMCDiagnosticTools
using Statistics: Statistics
using StatsBase: StatsBase

function MCMCDiagnosticTools._expectand_proxy(::typeof(StatsBase.mad), x)
    x_folded = MCMCDiagnosticTools._fold_around_median(x)
    return MCMCDiagnosticTools._expectand_proxy(Statistics.median, x_folded)
end

end
