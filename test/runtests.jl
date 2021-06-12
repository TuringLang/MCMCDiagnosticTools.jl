using Pkg

# Activate test environment on older Julia versions
@static if VERSION < v"1.2"
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=dirname(@__DIR__)))
    Pkg.instantiate()
end

using InferenceDiagnostics
using Test

@testset "InferenceDiagnostics.jl" begin
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
