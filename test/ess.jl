@testset "ess.jl" begin
    @testset "ESS and R̂ (IID samples)" begin
        rawx = randn(10_000, 10, 40)

        # Repeat tests with different scales
        for scale in (1, 50, 100)
            x = scale * rawx

            ess_standard, rhat_standard = ess_rhat(x)
            ess_standard2, rhat_standard2 = ess_rhat(x; method=ESSMethod())
            ess_fft, rhat_fft = ess_rhat(x; method=FFTESSMethod())
            ess_bda, rhat_bda = ess_rhat(x; method=BDAESSMethod())

            # check that we get (roughly) the same results
            @test ess_standard == ess_standard2
            @test ess_standard ≈ ess_fft
            @test rhat_standard == rhat_standard2 == rhat_fft == rhat_bda

            # check that the estimates are reasonable
            @test all(x -> isapprox(x, 100_000; rtol=0.1), ess_standard)
            @test all(x -> isapprox(x, 100_000; rtol=0.1), ess_bda)
            @test all(x -> isapprox(x, 1; rtol=0.1), rhat_standard)

            # BDA method fluctuates more
            @test var(ess_standard) < var(ess_bda)
        end
    end

    @testset "ESS and R̂ (identical samples)" begin
        x = ones(10_000, 10, 40)

        ess_standard, rhat_standard = ess_rhat(x)
        ess_standard2, rhat_standard2 = ess_rhat(x; method=ESSMethod())
        ess_fft, rhat_fft = ess_rhat(x; method=FFTESSMethod())
        ess_bda, rhat_bda = ess_rhat(x; method=BDAESSMethod())

        # check that the estimates are all NaN
        for ess in (ess_standard, ess_standard2, ess_fft, ess_bda)
            @test all(isnan, ess)
        end
        for rhat in (rhat_standard, rhat_standard2, rhat_fft, rhat_bda)
            @test all(isnan, rhat)
        end
    end

    @testset "ESS and R̂ (single sample)" begin # check that issue #137 is fixed
        x = rand(1, 3, 5)

        for method in (ESSMethod(), FFTESSMethod(), BDAESSMethod())
            # analyze array
            ess_array, rhat_array = ess_rhat(x; method=method)

            @test length(ess_array) == size(x, 3)
            @test all(ismissing, ess_array) # since min(maxlag, niter - 1) = 0
            @test length(rhat_array) == size(x, 3)
            @test all(ismissing, rhat_array)
        end
    end

    @testset "ESS and R̂ for chains with 2 epochs that have not mixed" begin
        # checks that splitting yields lower ESS estimates and higher Rhat estimates
        x = randn(1000, 4, 10) .+ repeat([0, 10]; inner=(500, 1, 1))
        ess_array, rhat_array = ess_rhat(x; split_chains=1)
        @test all(x -> isapprox(x, 1; rtol=0.1), rhat_array)
        ess_array2, rhat_array2 = ess_rhat(x; split_chains=2)
        @test all(ess_array2 .< ess_array)
        @test all(>(2), rhat_array2)
    end
end
