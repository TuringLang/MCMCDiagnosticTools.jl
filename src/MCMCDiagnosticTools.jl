module MCMCDiagnosticTools

using AbstractFFTs: AbstractFFTs
using DataAPI: DataAPI
using DataStructures: DataStructures
using Distributions: Distributions
using MLJModelInterface: MLJModelInterface as MMI
using SpecialFunctions: SpecialFunctions
using StatsBase: StatsBase
using StatsFuns: StatsFuns, sqrt2
using Tables: Tables

using LinearAlgebra: LinearAlgebra
using Random: Random
using Statistics: Statistics

export bfmi
export discretediag
export ess, ess_rhat, rhat, AutocovMethod, FFTAutocovMethod, BDAAutocovMethod
export gelmandiag, gelmandiag_multivariate
export gewekediag
export heideldiag
export mcse
export rafterydiag
export rstar

include("utils.jl")
include("bfmi.jl")
include("discretediag.jl")
include("ess_rhat.jl")
include("gelmandiag.jl")
include("gewekediag.jl")
include("heideldiag.jl")
include("mcse.jl")
include("rafterydiag.jl")
include("rstar.jl")
end
