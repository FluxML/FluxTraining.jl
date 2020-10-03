
using Pkg

Pkg.add("Publish")

using Publish
using FluxTraining

deploy(FluxTraining; root = "/FluxTraining.jl", label="dev", force = true)
