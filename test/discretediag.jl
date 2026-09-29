@testset "discretediag.jl" begin
    nparams = 4
    ndraws = 100
    nchains = 2
    samples = rand(-100:100, ndraws, nchains, nparams)

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

    @testset "transition_samplers" begin
        # state 2 was never observed to transition
        P = [0.2 0.8 0.0; 0.0 0.0 0.0; 0.5 0.0 0.5]
        samplers = MCMCDiagnosticTools.transition_samplers(P)
        @test length(samplers) == 3
        rng = Xoshiro(42)
        ndraws = 100_000
        for i in (1, 3)
            counts = zeros(Int, 3)
            for _ in 1:ndraws
                counts[rand(rng, samplers[i])] += 1
            end
            @test counts ./ ndraws ≈ P[i, :] atol = 0.01
        end
        @test all(rand(rng, samplers[2]) == 1 for _ in 1:100)
    end
end
