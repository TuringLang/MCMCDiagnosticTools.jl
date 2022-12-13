using MCMCDiagnosticTools
using Test

@testset "split_chains" begin
    x = rand(100, 4, 8)
    @test @inferred(MCMCDiagnosticTools.split_chains(x, 1)) == x
    @test MCMCDiagnosticTools.split_chains(x, 2) == reshape(x, 50, 8, 8)
    @test MCMCDiagnosticTools.split_chains(x, 3) == reshape(x[1:99, :, :], 33, 12, 8)
end
