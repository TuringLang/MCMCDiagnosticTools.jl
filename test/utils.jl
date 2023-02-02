using MCMCDiagnosticTools
using Test
using Random
using Statistics

@testset "unique_indices" begin
    @testset "indices=$(eachindex(inds))" for inds in [
        rand(11:14, 100), transpose(rand(11:14, 10, 10))
    ]
        unique, indices = @inferred MCMCDiagnosticTools.unique_indices(inds)
        @test unique isa Vector{Int}
        if eachindex(inds) isa CartesianIndices{2}
            @test indices isa Vector{Vector{CartesianIndex{2}}}
        else
            @test indices isa Vector{Vector{Int}}
        end
        @test issorted(unique)
        @test issetequal(union(indices...), eachindex(inds))
        for i in eachindex(unique, indices)
            @test all(inds[indices[i]] .== unique[i])
        end
    end
end

@testset "copy_split!" begin
    # check a matrix with even number of rows
    x = rand(50, 20)

    # check incompatible sizes
    @test_throws DimensionMismatch MCMCDiagnosticTools.copyto_split!(similar(x, 25, 20), x)
    @test_throws DimensionMismatch MCMCDiagnosticTools.copyto_split!(similar(x, 50, 40), x)

    y = similar(x, 25, 40)
    MCMCDiagnosticTools.copyto_split!(y, x)
    @test reshape(y, size(x)) == x

    # check a matrix with odd number of rows
    x = rand(51, 20)

    # check incompatible sizes
    @test_throws DimensionMismatch MCMCDiagnosticTools.copyto_split!(similar(x, 25, 20), x)
    @test_throws DimensionMismatch MCMCDiagnosticTools.copyto_split!(similar(x, 51, 40), x)

    MCMCDiagnosticTools.copyto_split!(y, x)
    @test reshape(y, 50, 20) == x[vcat(1:25, 27:51), :]

    # check with 3 splits
    y = similar(x, 16, 60)
    x = rand(50, 20)
    MCMCDiagnosticTools.copyto_split!(y, x)
    @test reshape(y, 48, :) == x[vcat(1:16, 18:33, 35:50), :]
    x = rand(49, 20)
    MCMCDiagnosticTools.copyto_split!(y, x)
    @test reshape(y, 48, :) == x[vcat(1:16, 18:33, 34:49), :]
end

@testset "split_chain_indices" begin
    c = [2, 2, 1, 3, 4, 3, 4, 1, 2, 1, 4, 3, 3, 2, 4, 3, 4, 1, 4, 1]
    @test @inferred(MCMCDiagnosticTools.split_chain_indices(c, 1)) == c

    cnew = @inferred MCMCDiagnosticTools.split_chain_indices(c, 2)
    @test issetequal(Base.unique(cnew), 1:maximum(cnew))  # check no indices skipped
    unique, indices = MCMCDiagnosticTools.unique_indices(c)
    uniquenew, indicesnew = MCMCDiagnosticTools.unique_indices(cnew)
    for (i, inew) in enumerate(1:2:7)
        @test length(indicesnew[inew]) ≥ length(indicesnew[inew + 1])
        @test indices[i] == vcat(indicesnew[inew], indicesnew[inew + 1])
    end

    cnew = MCMCDiagnosticTools.split_chain_indices(c, 3)
    @test issetequal(Base.unique(cnew), 1:maximum(cnew))  # check no indices skipped
    unique, indices = MCMCDiagnosticTools.unique_indices(c)
    uniquenew, indicesnew = MCMCDiagnosticTools.unique_indices(cnew)
    for (i, inew) in enumerate(1:3:11)
        @test length(indicesnew[inew]) ≥
            length(indicesnew[inew + 1]) ≥
            length(indicesnew[inew + 2])
        @test indices[i] ==
            vcat(indicesnew[inew], indicesnew[inew + 1], indicesnew[inew + 2])
    end
end

@testset "shuffle_split_stratified" begin
    rng = Random.default_rng()
    c = rand(1:4, 100)
    unique, indices = MCMCDiagnosticTools.unique_indices(c)
    @testset "frac=$frac" for frac in [0.3, 0.5, 0.7]
        inds1, inds2 = @inferred(MCMCDiagnosticTools.shuffle_split_stratified(rng, c, frac))
        @test issetequal(vcat(inds1, inds2), eachindex(c))
        for inds in indices
            common_inds = intersect(inds1, inds)
            @test length(common_inds) == round(frac * length(inds))
        end
    end
end

@testset "_rank_normalize" begin
    x = randexp(1000, 4, 8)
    z = @inferred MCMCDiagnosticTools._rank_normalize(x)
    @test size(z) == size(x)
    @test all(xi -> isapprox(xi, 0; atol=1e-13), mean(z; dims=(1, 2)))
    @test all(xi -> isapprox(xi, 1; rtol=1e-2), std(z; dims=(1, 2)))
end

@testset "_fold_around_median" begin
    x = rand(100, 4, 8)
    @inferred MCMCDiagnosticTools._fold_around_median(x)
    @test MCMCDiagnosticTools._fold_around_median(x) ≈ abs.(x .- median(x; dims=(1, 2)))
    x = Array{Union{Missing,Float64}}(undef, 100, 4, 8)
    x .= randn.()
    x[1, 1, 1] = missing
    foldx = @inferred(MCMCDiagnosticTools._fold_around_median(x))
    @test all(ismissing, foldx[:, :, 1])
    @test foldx[:, :, 2:end] ≈ abs.(x[:, :, 2:end] .- median(x[:, :, 2:end]; dims=(1, 2)))
end
