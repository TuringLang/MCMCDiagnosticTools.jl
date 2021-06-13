@testset "gelmandiag.jl" begin
    nparams = 4
    nchains = 2
    samples = randn(100, nparams, nchains)

    @testset "results" begin
        result = @inferred(gelmandiag(samples))
        @test result isa NamedTuple{(:psrf, :psrfci)}
        for name in (:psrf, :psrfci)
            x = getfield(result, name)
            @test x isa Vector{Float64}
            @test length(x) == nparams
        end

        result = @inferred(gelmandiag_multivariate(samples))
        @test result isa NamedTuple{(:psrf, :psrfci, :psrfmultivariate)}
        for name in (:psrf, :psrfci)
            x = getfield(result, name)
            @test x isa Vector{Float64}
            @test length(x) == nparams
        end
        @test result.psrfmultivariate isa Float64
    end

    @testset "exceptions" begin
        @test_throws ErrorException gelmandiag(samples[:, :, 1:1])
        @test_throws ErrorException gelmandiag_multivariate(samples[:, 1:1, :])
    end
end
