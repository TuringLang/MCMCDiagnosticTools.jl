using MCMCDiagnosticTools
using Test
using Random

@testset "indices_of_unique" begin
    inds = [1, 4, 3, 1, 4, 1, 3, 3, 4, 2, 1, 4, 1, 1, 3, 2, 3, 4, 4, 2]
    d = MCMCDiagnosticTools.indices_of_unique(inds)
    @test d isa Dict{Int, Vector{Int}}
    @test issetequal(union(values(d)...), eachindex(inds))
    for k in keys(d)
        @test all(inds[d[k]] .== k)
    end
end

@testset "split_chain_indices" begin
    c = [2, 2, 1, 3, 4, 3, 4, 1, 2, 1, 4, 3, 3, 2, 4, 3, 4, 1, 4, 1]
    MCMCDiagnosticTools.split_chain_indices(c, 1) == c

    cnew = MCMCDiagnosticTools.split_chain_indices(c, 2)
    d = MCMCDiagnosticTools.indices_of_unique(c)
    dnew = MCMCDiagnosticTools.indices_of_unique(cnew)
    for (i, inew) in enumerate(1:2:7)
        @test length(dnew[inew]) ≥ length(dnew[inew + 1])
        @test d[i] == vcat(dnew[inew], dnew[inew + 1])
    end

    cnew = MCMCDiagnosticTools.split_chain_indices(c, 3)
    d = MCMCDiagnosticTools.indices_of_unique(c)
    dnew = MCMCDiagnosticTools.indices_of_unique(cnew)
    for (i, inew) in enumerate(1:3:11)
        @test length(dnew[inew]) ≥ length(dnew[inew + 1]) ≥ length(dnew[inew + 2])
        @test d[i] == vcat(dnew[inew], dnew[inew + 1], dnew[inew + 2])
    end
end

@testset "shuffle_split_stratified" begin
    rng = Random.default_rng()
    c = rand(1:4, 100)
    d = MCMCDiagnosticTools.indices_of_unique(c)
    @testset "frac=$frac" for frac in [0.3, 0.5, 0.7]
        inds1, inds2 = MCMCDiagnosticTools.shuffle_split_stratified(rng, c, frac)
        @test issetequal(vcat(inds1, inds2), eachindex(c))
        for i in 1:4
            common_inds = intersect(inds1, d[i])
            @test length(common_inds) == round(frac * length(d[i]))
        end
    end
end
