using MCMCDiagnosticTools
using Aqua
using Test

@testset "Aqua" begin
    # Test ambiguities separately without Base and Core
    # Ref: https://github.com/JuliaTesting/Aqua.jl/issues/77
    persistent_tasks = parse(Bool, get(ENV, "AQUA_PERSISTENT_TASKS", "true"))
    Aqua.test_all(MCMCDiagnosticTools; ambiguities=false, persistent_tasks)
    Aqua.test_ambiguities(MCMCDiagnosticTools)
end
