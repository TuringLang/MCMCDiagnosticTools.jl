using MCMCDiagnosticTools
using Aqua
using Test

@testset "Aqua" begin
    # Test ambiguities separately without Base and Core
    # Ref: https://github.com/JuliaTesting/Aqua.jl/issues/77
    # Only test Project.toml formatting on Julia > 1.6 when running Github action
    # Ref: https://github.com/JuliaTesting/Aqua.jl/issues/105
    Aqua.test_all(
        MCMCDiagnosticTools;
        ambiguities=false,
        project_toml_formatting=VERSION >= v"1.7" || !haskey(ENV, "GITHUB_ACTIONS"),
    )
    Aqua.test_ambiguities([MCMCDiagnosticTools])
end
