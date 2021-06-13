@testset "heideldiag.jl" begin
    samples = randn(100)

    @testset "results" begin
        @test @inferred(heideldiag(samples)) isa NamedTuple
    end
end
