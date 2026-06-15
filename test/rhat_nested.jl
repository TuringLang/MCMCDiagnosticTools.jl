using MCMCDiagnosticTools
using OffsetArrays
using Statistics
using Test

# Generate AR(1) chains grouped into superchains. All chains within the same superchain
# share the same value from init_vals, while stationary distribution is N(0, σ²/(1-φ²)).
function ar1_superchains(
    φ::Real, σ::Real, ndraws::Int, nchains_per_init::Int, init_vals::AbstractArray
)
    (nsuperchains, nparams...) = size(init_vals)
    nchains = nsuperchains * nchains_per_init
    x_init = repeat(init_vals; inner=(nchains_per_init, map(one, nparams)...))
    x = randn(eltype(x_init), ndraws, nchains, nparams...)
    x .*= σ
    @views copyto!(x[1, :, map(_ -> :, nparams)...], x_init)
    accumulate!(x, x; dims=1) do xi, ϵ
        return muladd(φ, xi, ϵ)
    end
    return x
end

@testset "rhat_nested.jl" begin
    @testset "basics" begin
        # sizes cover 0, 1, and 2 parameter axes; eltypes cover Float32, Float64, Int
        @testset "promotes eltype when necessary" begin
            ids = [1, 1, 2, 2]
            sizes = ((100, 4), (100, 4, 2), (100, 4, 2, 3))

            @testset for kind in (:rank, :bulk, :tail, :basic), sz in sizes
                @testset for T in (Float32, Float64, Int)
                    x = T <: Int ? rand(1:10, sz...) : rand(T, sz...)
                    TV = length(sz) < 3 ? float(T) : Array{float(T),length(sz) - 2}
                    @test @inferred(rhat_nested(x, ids; kind)) isa TV
                end
            end
        end

        @testset "errors" begin
            x = rand(100, 4)
            ids = [1, 1, 2, 2]
            @testset "requires at least 2D input" begin
                @test_throws ArgumentError rhat_nested(rand(100), [1])
            end
            @testset "superchain_ids shorted than nchains" begin
                @test_throws DimensionMismatch rhat_nested(x, [1, 1, 2])
            end
            @testset "superchain_ids longer than nchains" begin
                @test_throws DimensionMismatch rhat_nested(x, [1, 1, 2, 2, 3])
            end
            @testset "requires at least 2 superchains" begin
                @test_throws ArgumentError rhat_nested(x, [1, 1, 1, 1])
            end
            @testset "requires equal-sized superchains" begin
                @test_throws ArgumentError rhat_nested(x, [1, 1, 1, 2])
            end
            @testset "unknown kind" begin
                @test_throws ArgumentError rhat_nested(x, ids; kind=:foo)
            end
        end

        @testset "Union{Missing,Float64} eltype" begin
            @testset for kind in (:rank, :bulk, :tail, :basic)
                x = Array{Union{Missing,Float64}}(undef, 1000, 4, 3)
                x .= randn.()
                x[1, 1, 1] = missing
                R = rhat_nested(x, [1, 1, 2, 2]; kind)
                @test ismissing(R[1])
                @test !any(ismissing, R[2:3])
            end
        end

        @testset "produces similar arrays to inputs" begin
            ids = [1, 1, 2, 2]
            @testset for kind in (:rank, :bulk, :tail, :basic),
                _axes in ((-5:94, 2:5, 11:15), (-5:94, 2:5, 11:15, 0:2))

                x = randn(map(length, _axes)...)
                N = ndims(x)
                y = OffsetArray(x, _axes...)
                R = rhat_nested(y, ids; kind)
                @test R isa OffsetArray{Float64,N - 2}
                @test axes(R) == _axes[3:end]
                # plain array gives same values
                @test rhat_nested(x, ids; kind) == collect(R)
                # all-Missing input
                z = OffsetArray(similar(x, Missing), _axes...)
                Rm = rhat_nested(z, ids; kind)
                @test Rm isa OffsetArray{Missing,N - 2}
                @test axes(Rm) == _axes[3:end]
            end
        end

        @testset "scalar output for 2D input" begin
            @testset for kind in (:rank, :bulk, :tail, :basic)
                @test @inferred(rhat_nested(randn(100, 4), [1, 1, 2, 2]; kind)) isa Float64
            end
        end
    end

    @testset "correctness" begin
        @testset "invariant to superchain label type and values" begin
            x = randn(100, 4, 10)
            ids_int = [1, 1, 2, 2]
            ids_big = [42, 42, 99, 99]
            ids_char = ['a', 'a', 'b', 'b']
            @testset for kind in (:rank, :bulk, :tail, :basic)
                R = rhat_nested(x, ids_int; kind)
                @test R == rhat_nested(x, ids_big; kind) == rhat_nested(x, ids_char; kind)
            end
        end

        @testset "invariant to simultaneous permutation of chains and superchain_ids" begin
            @testset for nsuperchains in (8, 16),
                nchains_per_superchain in (2, 4),
                split_chains in (1, 2)

                nchains = nsuperchains * nchains_per_superchain
                x = randn(100, nchains, 10)
                superchain_ids = repeat(1:nsuperchains; inner=nchains_per_superchain)
                perm = randperm(nchains)
                @testset for kind in (:rank, :bulk, :tail, :basic)
                    R = rhat_nested(x, superchain_ids; kind, split_chains)
                    R_perm = rhat_nested(
                        x[:, perm, :], superchain_ids[perm]; kind, split_chains
                    )
                    @test R == R_perm
                end
            end
        end

        @testset "consistency with rhat (1 chain per superchain, no splitting)" begin
            # With one chain per superchain and no splitting the shared B/W terms give:
            #   rhat_nested² = 1       + B/W
            #   rhat²        = (n-1)/n + B/W
            @testset for kind in (:basic, :bulk, :tail),
                ndraws in (10, 20),
                nchains in (4, 8)

                x = randn(ndraws, nchains, 10)
                superchain_ids = collect(1:nchains)
                Rn = rhat_nested(x, superchain_ids; kind, split_chains=1)
                Rs = rhat(x; kind, split_chains=1)
                @test Rn ≈ @. sqrt(Rs^2 + (1 // ndraws))
            end
        end

        @testset ":rank == max(:bulk, :tail)" begin
            x = randn(100, 4, 10)
            superchain_ids = [1, 1, 2, 2]
            Rbulk = rhat_nested(x, superchain_ids; kind=:bulk)
            Rtail = rhat_nested(x, superchain_ids; kind=:tail)
            Rrank = rhat_nested(x, superchain_ids; kind=:rank)
            @test Rrank == max.(Rbulk, Rtail)
        end

        @testset "IID samples → nested R̂ ≈ 1" begin
            @testset for nsuperchains in (8, 16)
                nchains = 2048
                nchains_per_superchain = nchains ÷ nsuperchains
                nparams = 32
                superchain_ids = repeat(1:nsuperchains; inner=nchains_per_superchain)
                x = ar1(0, 1, 10, nchains, nparams)
                R = rhat_nested(x, superchain_ids)
                Rmin, Rmax = extrema(R)
                @test Rmin > 1
                @test Rmax < 1.001
            end
        end

        @testset "detects (non-)convergence" begin
            @testset "$name" for (name, φ, ndraws_warmup, (Rmin_exp, Rmax_exp)) in [
                ("short warm-up, low autocorrelation", 0.1, 10, (1, 1.01)),
                ("short warm-up, high autocorrelation", 0.9, 10, (1.01, Inf)),
                ("long warm-up, high autocorrelation", 0.9, 100, (0, 1.01)),
            ]
                σ = sqrt(1 - φ^2)
                nsuperchains = 8
                nchains_per_superchain = 2048 ÷ nsuperchains
                ndraws = 16
                nparams = 32
                superchain_ids = repeat(1:nsuperchains; inner=nchains_per_superchain)
                init_vals = randn(nsuperchains, nparams) * 100
                x = ar1_superchains(
                    φ, σ, ndraws_warmup + ndraws, nchains_per_superchain, init_vals
                )
                x_draws = x[(ndraws_warmup + 1):end, :, :]
                R = rhat_nested(x_draws, superchain_ids)
                @test all(isfinite, R)
                Rmin, Rmax = extrema(R)
                @test Rmin > Rmin_exp
                @test Rmax < Rmax_exp
            end
        end
    end
end
