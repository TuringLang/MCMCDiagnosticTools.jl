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
    # Write your tests here.
end
