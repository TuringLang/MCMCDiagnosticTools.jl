@testset "gewekediag.jl" begin
    @testset "results" begin
        @testset for T in (Float32, Float64)
            samples = randn(T, 100)
            @inferred NamedTuple{(:zscore, :pvalue),Tuple{T,T}} gewekediag(samples)
        end
    end

    @testset "exceptions" begin
        samples = randn(100)
        for x in (-0.3, 0, 1, 1.2)
            @test_throws ArgumentError gewekediag(samples; first=x)
            @test_throws ArgumentError gewekediag(samples; last=x)
        end
        @test_throws ArgumentError gewekediag(samples; first=0.6, last=0.5)
    end
end
