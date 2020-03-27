using FluxTraining
using Documenter

makedocs(;
    modules=[FluxTraining],
    authors="lorenzoh <lorenz.ohly@gmail.com>",
    repo="https://github.com/lorenzoh/FluxTraining.jl/blob/{commit}{path}#L{line}",
    sitename="FluxTraining.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lorenzoh.github.io/FluxTraining.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lorenzoh/FluxTraining.jl",
)
