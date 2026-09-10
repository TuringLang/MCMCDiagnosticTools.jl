using FillArrays: FillArrays
using MCMCDiagnosticTools: _Samples, _eachparam, _allocate_split_samples, copyto_split!
using OffsetArrays
using Test

@testset "_Samples" begin
    @testset "dense input is reshaped without copying" begin
        for shape in ((17,), (17, 3), (17, 3, 2), (17, 3, 2, 4))
            x = reshape(collect(1:prod(shape)), shape)
            samples = @inferred _Samples(x)
            @test samples.prototype === x
            @test samples.param_axes == axes(x)[3:end]
            @test size(samples.data) == (size(x, 1) * size(x, 2), prod(shape[3:end]))
            @test Base.mightalias(samples.data, x)
            @test collect(samples.lengths) == fill(size(x, 1), size(x, 2))
            @test samples.lengths isa FillArrays.AbstractFillVector
            @test vec(samples.data) == vec(x)
        end
    end

    @testset "ragged input is concatenated along draws" begin
        for shape in ((), (2,), (2, 3))
            x = [reshape(collect(1:(n * prod(shape))), n, shape...) for n in (17, 26, 41)]
            samples = @inferred _Samples(x)
            @test samples.prototype === first(x)
            @test samples.param_axes == axes(first(x))[2:end]
            @test size(samples.data) == (84, prod(shape))
            @test samples.lengths == [17, 26, 41]
            @test samples.lengths isa Vector
            @test samples.data == reshape(reduce(vcat, x), 84, :)
        end
    end

    @testset "noncontiguous and offset input" begin
        parent = reshape(collect(1:480), 20, 3, 2, 4)
        x = view(parent, 1:2:20, :, :, 2:3)
        for input in (x, OffsetArray(x, -4:5, 0:2, 5:6, 8:9))
            samples = _Samples(input)
            @test Base.mightalias(samples.data, parent)
            @test samples.data == reshape(collect(input), 30, 4)
            @test samples.param_axes == axes(input)[3:end]
        end
        ragged = [view(x, 1:n, i, :, :) for (i, n) in enumerate((7, 9, 10))]
        @test _Samples(ragged).data == reshape(reduce(vcat, ragged), 26, 4)
    end

    @testset "split workspace is reusable and leaves input intact" begin
        x = reshape(collect(1:276), 23, 3, 4)
        original = copy(x)
        for split in (1, 2, 3, 5)
            samples = _Samples(x)
            workspace = _allocate_split_samples(samples, Float64, split)
            reference = similar(workspace)
            for (param, pooled) in enumerate(_eachparam(samples))
                copyto_split!(workspace, pooled, samples.lengths)
                copyto_split!(reference, view(x, :, :, param))
                @test workspace == reference
            end
        end
        @test x == original
    end

    @testset "ragged splits retain every draw in order" begin
        for lengths in ((43, 66, 101), (43, 43, 43), (0, 1, 2, 7)), split in (1, 2, 3, 5)
            x = [reshape(collect(1:(n * 2)), n, 2) for n in lengths]
            original = deepcopy(x)
            samples = _Samples(x)
            workspace = @inferred _allocate_split_samples(samples, Float64, split)
            @test length(workspace) == length(lengths) * split
            @test sum(length, workspace) == sum(lengths)
            for (n, buffers) in zip(lengths, Iterators.partition(workspace, split))
                sizes = length.(buffers)
                @test sum(sizes) == n
                @test maximum(sizes) - minimum(sizes) ≤ 1
                @test issorted(sizes; rev=true) # Remainder draws go to the first splits.
            end
            for (param, pooled) in enumerate(_eachparam(samples))
                # The pooled draw vector need not be one-based either.
                for input in (pooled, OffsetArray(pooled, -3:(length(pooled) - 4)))
                    copyto_split!(workspace, input, samples.lengths)
                    @test reduce(vcat, vec.(workspace)) == collect(pooled)
                    for (chain, buffers) in zip(x, Iterators.partition(workspace, split))
                        @test reduce(vcat, vec.(buffers)) == chain[:, param]
                    end
                end
            end
            @test x == original
        end
    end

    @testset "empty parameter axes" begin
        @test size(_Samples(zeros(10, 2, 0)).data) == (20, 0)
        @test size(_Samples([zeros(10, 0), zeros(12, 0)]).data) == (22, 0)
    end
end
