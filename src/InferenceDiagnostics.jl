module InferenceDiagnostics

import DataAPI
import Distributions
import MLJModelInterface
import StatsBase
import Tables

import LinearAlgebra
import Random
import Statistics

export discretediag
export rstar

include("discretediag.jl")
include("rstar.jl")
end
