@testset "heideldiag.jl" begin
    @testset "results" begin
        @testset for T in (Float32, Float64)
            samples = randn(T, 100)
            @test @inferred(heideldiag(samples)) isa NamedTuple
        end
    end
end
