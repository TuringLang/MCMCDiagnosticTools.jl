@testset "rafterydiag.jl" begin
    samples = randn(5_000)

    @testset "results" begin
        @test @inferred(rafterydiag(samples)) isa NamedTuple
    end

    @testset "warning" begin
        @test_logs (:warn,) rafterydiag(samples[1:100])
    end
end
