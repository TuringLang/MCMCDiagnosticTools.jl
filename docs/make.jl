using Documenter
using DocumenterInterLinks

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using MCMCDiagnosticTools

DocMeta.setdocmeta!(
    MCMCDiagnosticTools, :DocTestSetup, :(using MCMCDiagnosticTools); recursive=true
)

links = InterLinks(
    "MLJ" => "https://juliaai.github.io/MLJ.jl/stable/",
    "Statistics" => "https://docs.julialang.org/en/v1/",
    "StatsBase" => (
        "https://juliastats.org/StatsBase.jl/stable/",
        "https://juliastats.org/StatsBase.jl/dev/objects.inv",
    ),
)

makedocs(;
    modules=[MCMCDiagnosticTools],
    authors="David Widmann",
    repo=Remotes.GitHub("TuringLang", "MCMCDiagnosticTools.jl"),
    sitename="MCMCDiagnosticTools.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://turinglang.github.io/MCMCDiagnosticTools.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
    warnonly=:footnote,
    checkdocs=:exports,
    plugins=[links],
)
