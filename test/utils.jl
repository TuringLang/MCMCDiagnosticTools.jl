using MCMCDiagnosticTools
using Test

@testset "indices_of_unique" begin
    inds = [1, 4, 3, 1, 4, 1, 3, 3, 4, 2, 1, 4, 1, 1, 3, 2, 3, 4, 4, 2]
    d = MCMCDiagnosticTools.indices_of_unique(inds)
    @test d isa Dict{Int, Vector{Int}}
    @test issetequal(union(values(d)...), eachindex(inds))
    for k in keys(d)
        @test all(inds[d[k]] .== k)
    end
end
