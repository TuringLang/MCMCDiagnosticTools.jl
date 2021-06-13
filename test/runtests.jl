using Pkg

# Activate test environment on older Julia versions
@static if VERSION < v"1.2"
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=dirname(@__DIR__)))
    Pkg.instantiate()
end

using InferenceDiagnostics
using FFTW

using Random
using Statistics
using Test

Random.seed!(1)

@testset "InferenceDiagnostics.jl" begin
    @testset "discrete diagnostic" begin include("discretediag.jl") end
    @testset "ESS" begin include("ess.jl") end
    @testset "Gelman, Rubin and Brooks diagnostic" begin include("gelmandiag.jl") end
    @testset "Geweke diagnostic" begin include("gewekediag.jl") end
    @testset "Heidelberger and Welch diagnostic" begin include("heideldiag.jl") end
    @testset "Monte Carlo standard error" begin include("mcse.jl") end
    @testset "Raftery and Lewis diagnostic" begin include("rafterydiag.jl") end
    @testset "R⋆ diagnostic" begin
        # MLJXGBoostInterface requires Julia >= 1.3
        # XGBoost errors on 32bit systems: https://github.com/dmlc/XGBoost.jl/issues/92
        if VERSION >= v"1.3" && Sys.WORD_SIZE == 64
            # run tests related to rstar statistic
            Pkg.activate("rstar")
            Pkg.develop(; path=dirname(dirname(pathof(InferenceDiagnostics))))
            Pkg.instantiate()
            include(joinpath("rstar", "runtests.jl"))
        else
            @info "R⋆ not tested: requires Julia >= 1.3 and a 64bit architecture"
        end
    end
end
