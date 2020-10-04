
using Pkg

Pkg.add("Publish")
Pkg.add("GraphPlot")

using Publish
using FluxTraining

deploy(FluxTraining; root = "/FluxTraining.jl", label="dev", force = true)
