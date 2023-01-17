using MCMCDiagnosticTools
using FFTW

using Random
using Statistics
using Test

Random.seed!(1)

@testset "MCMCDiagnosticTools.jl" begin
    include("helpers.jl")

    @testset "utils" begin
        include("utils.jl")
    end

    @testset "Bayesian fraction of missing information" begin
        include("bfmi.jl")
    end

    @testset "discrete diagnostic" begin
        include("discretediag.jl")
    end
    @testset "ESS" begin
        include("ess.jl")
    end
    @testset "Monte Carlo standard error" begin
        include("mcse.jl")
    end
    @testset "Gelman, Rubin and Brooks diagnostic" begin
        include("gelmandiag.jl")
    end
    @testset "Geweke diagnostic" begin
        include("gewekediag.jl")
    end
    @testset "Heidelberger and Welch diagnostic" begin
        include("heideldiag.jl")
    end
    @testset "Raftery and Lewis diagnostic" begin
        include("rafterydiag.jl")
    end
    @testset "R⋆ diagnostic" begin
        include("rstar.jl")
    end
end
