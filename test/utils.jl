using MCMCDiagnosticTools
using Test
using OffsetArrays
using Random
using Statistics
using StatsBase: StatsBase

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
    @testset for sz in ((1000,), (1000, 4), (1000, 4, 8), (1000, 4, 8, 2))
        x = randexp(sz...)
        dims = MCMCDiagnosticTools._sample_dims(x)
        z = @inferred MCMCDiagnosticTools._rank_normalize(x)
        @test size(z) == size(x)
        @test all(xi -> isapprox(xi, 0; atol=1e-13), mean(z; dims))
        @test all(xi -> isapprox(xi, 1; rtol=1e-2), std(z; dims))
    end
end

# RNG that always generates the same `Float64`, used to control `_wsample`
struct FixedRNG <: Random.AbstractRNG
    x::Float64
end
function Random.rand(rng::FixedRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64}})
    return rng.x
end

@testset "_wsample" begin
    @testset "basic" begin
        w = [0.2, 0.0, 0.5, 0.3]
        # the first index whose cumulative weight is at least `rand() * sum(w)` is sampled
        @test MCMCDiagnosticTools._wsample(FixedRNG(0.0), w) == 1
        @test MCMCDiagnosticTools._wsample(FixedRNG(0.1), w) == 1
        @test MCMCDiagnosticTools._wsample(FixedRNG(0.3), w) == 3
        @test MCMCDiagnosticTools._wsample(FixedRNG(0.9), w) == 4
        @test @inferred(MCMCDiagnosticTools._wsample(Random.default_rng(), w)) isa Int
    end

    @testset "all-zero weights return the first index" begin
        @test MCMCDiagnosticTools._wsample(FixedRNG(0.5), zeros(3)) == 1
    end

    @testset "never returns an index with zero weight" begin
        # emulate floating-point error causing `rand() * sum(w)` to exceed the cumulative
        # sum of the weights
        w = [0.2, 0.5, 0.3, 0.0, 0.0]
        rng = FixedRNG(nextfloat(1.0))
        @test MCMCDiagnosticTools._wsample(rng, w) == 3
        @test MCMCDiagnosticTools._wsample(rng, w) ==
            StatsBase.wsample(rng, eachindex(w), w)
    end

    @testset "offset indices" begin
        w = OffsetArray([0.2, 0.0, 0.5, 0.3], -2)
        @test MCMCDiagnosticTools._wsample(FixedRNG(0.0), w) == -1
        @test MCMCDiagnosticTools._wsample(FixedRNG(0.3), w) == 1
    end

    @testset "consistent with StatsBase.wsample" begin
        p = rand(10)
        p ./= sum(p)
        weights = (
            [1.0],
            rand(5),
            [rand(5); zeros(3)],
            [0.0; rand(3); 0.0; rand(2)],
            p,
            rand(0:3, 8),
        )
        for w in weights
            seed = rand(UInt)
            rng1, rng2 = Xoshiro(seed), Xoshiro(seed)
            @test all(1:100) do _
                return MCMCDiagnosticTools._wsample(rng1, w) ==
                       StatsBase.wsample(rng2, eachindex(w), w)
            end
        end
    end

    @testset "sampling frequencies" begin
        w = [1.0, 0.0, 3.0, 6.0]
        rng = Xoshiro(42)
        ndraws = 100_000
        counts = zeros(Int, length(w))
        for _ in 1:ndraws
            counts[MCMCDiagnosticTools._wsample(rng, w)] += 1
        end
        @test counts[2] == 0
        @test counts ./ ndraws ≈ w ./ sum(w) atol = 0.01
    end
end

@testset "_fold_around_median" begin
    @testset for sz in ((1000,), (1000, 4), (1000, 4, 8), (1000, 4, 8, 2))
        x = rand(sz...)
        dims = MCMCDiagnosticTools._sample_dims(x)
        @inferred MCMCDiagnosticTools._fold_around_median(x)
        @test MCMCDiagnosticTools._fold_around_median(x) ≈ abs.(x .- median(x; dims))
        x = Array{Union{Missing,Float64}}(undef, sz...)
        x .= randn.()
        x[1] = missing
        foldx = @inferred(MCMCDiagnosticTools._fold_around_median(x))
        @test all(ismissing, foldx[:, :, 1, 1])
        length(sz) > 2 && @test foldx[:, :, 2:end, :] ≈
            abs.(x[:, :, 2:end, :] .- median(x[:, :, 2:end, :]; dims))
    end
end

@testset "_sample_dims" begin
    x = randn(10)
    @test @inferred(MCMCDiagnosticTools._sample_dims(x)) === (1,)
    x = randn(10, 2)
    @test @inferred(MCMCDiagnosticTools._sample_dims(x)) === (1, 2)
    x = randn(10, 2, 3)
    @test @inferred(MCMCDiagnosticTools._sample_dims(x)) === (1, 2)
    x = randn(10, 2, 3, 4)
    @test @inferred(MCMCDiagnosticTools._sample_dims(x)) === (1, 2)
end

@testset "_param_dims" begin
    x = randn(10)
    @test @inferred(MCMCDiagnosticTools._param_dims(x)) === ()
    x = randn(10, 2)
    @test @inferred(MCMCDiagnosticTools._param_dims(x)) === ()
    x = randn(10, 2, 3)
    @test @inferred(MCMCDiagnosticTools._param_dims(x)) === (3,)
    x = randn(10, 2, 3, 4)
    @test @inferred(MCMCDiagnosticTools._param_dims(x)) === (3, 4)
end

@testset "_param_axes" begin
    x = OffsetArray(randn(10), -4:5)
    @test @inferred(MCMCDiagnosticTools._param_axes(x)) === ()
    x = OffsetArray(randn(10, 2), -4:5, 0:1)
    @test @inferred(MCMCDiagnosticTools._param_axes(x)) === ()
    x = OffsetArray(randn(10, 2, 3), -4:5, 0:1, -3:-1)
    @test @inferred(MCMCDiagnosticTools._param_axes(x)) === (axes(x, 3),)
    x = OffsetArray(randn(10, 2, 3, 4), -4:5, 0:1, -3:-1, 0:3)
    @test @inferred(MCMCDiagnosticTools._param_axes(x)) === (axes(x, 3), axes(x, 4))
end

@testset "_params_array" begin
    x = randn(10)
    @test MCMCDiagnosticTools._params_array(x) == reshape(x, :, 1, 1)
    @test MCMCDiagnosticTools._params_array(x, 1) == x
    @test MCMCDiagnosticTools._params_array(x, 2) == reshape(x, :, 1)
    @test MCMCDiagnosticTools._params_array(x, 3) == reshape(x, :, 1, 1)
    @test MCMCDiagnosticTools._params_array(x, 4) == reshape(x, :, 1, 1, 1)
    x = randn(10, 2)
    @test MCMCDiagnosticTools._params_array(x) == reshape(x, size(x)..., 1)
    @test MCMCDiagnosticTools._params_array(x, 1) == vec(x)
    @test MCMCDiagnosticTools._params_array(x, 2) == x
    @test MCMCDiagnosticTools._params_array(x, 3) == reshape(x, size(x)..., 1)
    @test MCMCDiagnosticTools._params_array(x, 4) == reshape(x, size(x)..., 1, 1)
    x = randn(10, 2, 3)
    @test MCMCDiagnosticTools._params_array(x) == x
    @test MCMCDiagnosticTools._params_array(x, 1) == vec(x)
    @test MCMCDiagnosticTools._params_array(x, 2) == reshape(x, size(x, 1), :)
    @test MCMCDiagnosticTools._params_array(x, 3) == x
    @test MCMCDiagnosticTools._params_array(x, 4) == reshape(x, size(x)..., 1)
    x = randn(10, 2, 3, 4)
    @test MCMCDiagnosticTools._params_array(x) == reshape(x, size(x, 1), size(x, 2), :)
    @test MCMCDiagnosticTools._params_array(x, 1) == vec(x)
    @test MCMCDiagnosticTools._params_array(x, 2) == reshape(x, size(x, 1), :)
    @test MCMCDiagnosticTools._params_array(x, 3) == reshape(x, size(x, 1), size(x, 2), :)
    @test MCMCDiagnosticTools._params_array(x, 4) == x

    @test_throws ArgumentError MCMCDiagnosticTools._params_array(x, -1)
    @test_throws ArgumentError MCMCDiagnosticTools._params_array(x, 0)
end

@testset "_maybescalar" begin
    sz = (1, 2, 3)
    @testset for d in 0:length(sz)
        x = randn(sz[1:d])
        if d == 0
            @test MCMCDiagnosticTools._maybescalar(x) === x[]
        else
            @test MCMCDiagnosticTools._maybescalar(x) === x
        end
    end
end
