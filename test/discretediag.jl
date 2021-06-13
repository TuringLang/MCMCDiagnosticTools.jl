@testset "discretediag.jl" begin
    nparams = 4
    nchains = 2
    samples = rand(-100:100, 100, nparams, nchains)

    @testset "results" begin
        for method in
            (:weiss, :hangartner, :DARBOOT, :MCBOOT, :billingsley, :billingsleyBOOT)
            between_chain, within_chain = @inferred(discretediag(samples; method=method))

            @test between_chain isa NamedTuple{(:stat, :df, :pvalue)}
            for name in (:stat, :df, :pvalue)
                x = getfield(between_chain, name)
                @test x isa Vector{Float64}
                @test length(x) == nparams
            end

            @test within_chain isa NamedTuple{(:stat, :df, :pvalue)}
            for name in (:stat, :df, :pvalue)
                x = getfield(within_chain, name)
                @test x isa Matrix{Float64}
                @test size(x) == (nparams, nchains)
            end
        end
    end

    @testset "exceptions" begin
        @test_throws ArgumentError discretediag(samples; method=:somemethod)
        for x in (-0.3, 0, 1, 1.2)
            @test_throws ArgumentError discretediag(samples; frac=x)
        end
    end
end
