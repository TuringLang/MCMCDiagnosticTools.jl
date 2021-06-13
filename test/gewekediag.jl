@testset "gewekediag.jl" begin
    samples = randn(100)

    @testset "results" begin
        @test @inferred(gewekediag(samples)) isa
              NamedTuple{(:zscore, :pvalue),Tuple{Float64,Float64}}
    end

    @testset "exceptions" begin
        for x in (-0.3, 0, 1, 1.2)
            @test_throws ArgumentError gewekediag(samples; first=x)
            @test_throws ArgumentError gewekediag(samples; last=x)
        end
        @test_throws ArgumentError gewekediag(samples; first=0.6, last=0.5)
    end
end
