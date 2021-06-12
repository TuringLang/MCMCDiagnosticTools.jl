@testset "mcse.jl" begin
    samples = randn(100)

    @testset "results" begin
        result = @inferred(mcse(samples))
        @test result isa Float64
        @test result > 0

        for method in (:imse, :ipse, :bm)
            result = @inferred(mcse(samples; method=method))
            @test result isa Float64
            @test result > 0
        end
    end

    @testset "warning" begin
        for size in (51, 75, 100, 153)
            @test_logs (:warn,) mcse(samples; method=:bm, size=size)
        end
    end

    @testset "exception" begin
        @test_throws ArgumentError mcse(samples; method=:somemethod)
    end
end
