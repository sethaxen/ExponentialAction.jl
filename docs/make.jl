using ExponentialAction
using Documenter

makedocs(;
    modules=[ExponentialAction],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo="https://github.com/sethaxen/ExponentialAction.jl/blob/{commit}{path}#L{line}",
    sitename="ExponentialAction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sethaxen.github.io/ExponentialAction.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/sethaxen/ExponentialAction.jl")
