@testset "bfmi.jl" begin
    energy = [1, 2, 3, 4]
    @test @inferred(bfmi(energy)) isa Real
    @test bfmi(energy) ≈ 0.6

    # energy values derived from sampling a 10-dimensional Cauchy
    energy = [
        42, 44, 45, 46, 42, 43, 36, 36, 31, 36, 36, 32, 36, 31, 31, 29, 29, 30, 25, 26, 29,
        29, 27, 30, 31, 29
    ]
    @test bfmi(energy) ≈ 0.2406937229

    energy_multichain = repeat(energy, 1, 4)
    @test @inferred(bfmi(energy_multichain)) isa Vector{<:Real}
    @test bfmi(energy_multichain) ≈ fill(0.2406937229, 4)
    @test bfmi(energy_multichain) ≈ bfmi(energy_multichain'; dims=2)
end
