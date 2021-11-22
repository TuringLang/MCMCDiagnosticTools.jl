using Documenter

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using MCMCDiagnosticTools

DocMeta.setdocmeta!(
    MCMCDiagnosticTools, :DocTestSetup, :(using MCMCDiagnosticTools); recursive=true
)

makedocs(;
    modules=[MCMCDiagnosticTools],
    authors="David Widmann",
    repo="https://github.com/TuringLang/MCMCDiagnosticTools.jl/blob/{commit}{path}#{line}",
    sitename="MCMCDiagnosticTools.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://turinglang.github.io/MCMCDiagnosticTools.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/TuringLang/MCMCDiagnosticTools.jl", push_preview=true, devbranch="main"
)
