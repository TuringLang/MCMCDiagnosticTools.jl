@testset "mcse.jl" begin
    @testset "results 1d" begin
        samples = randn(100)
        result = @inferred(mcse(samples))
        @test result isa Float64
        @test result > 0

        for method in (:imse, :ipse, :bm)
            result = @inferred(mcse(samples; method=method))
            @test result isa Float64
            @test result > 0
        end
    end

    @testset  "results 3d" begin
        nparams = 2
        nchains = 4
        samples = randn(nparams, 100, nchains)
        result = mcse(samples)  # mapslices is not type-inferrable
        @test result isa Vector{Float64}
        @test length(result) == nparams
        @test all(r -> r > 0, result)

        for method in (:imse, :ipse, :bm)
            result = mcse(samples)  # mapslices is not type-inferrable
            @test result isa Vector{Float64}
            @test length(result) == nparams
            @test all(r -> r > 0, result)
        end
    end

    @testset "warning" begin
        samples = randn(100)
        for size in (51, 75, 100, 153)
            @test_logs (:warn,) mcse(samples; method=:bm, size=size)
        end
    end

    @testset "exception" begin
        samples = randn(100)
        @test_throws ArgumentError mcse(samples; method=:somemethod)
    end
end
