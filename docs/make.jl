using Documenter

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using InferenceDiagnostics

DocMeta.setdocmeta!(
    InferenceDiagnostics, :DocTestSetup, :(using InferenceDiagnostics); recursive=true
)

makedocs(;
    modules=[InferenceDiagnostics],
    authors="David Widmann",
    repo="https://github.com/devmotion/InferenceDiagnostics.jl/blob/{commit}{path}#{line}",
    sitename="InferenceDiagnostics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://devmotion.github.io/InferenceDiagnostics.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/devmotion/InferenceDiagnostics.jl", push_preview=true, devbranch="main"
)
