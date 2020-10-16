using Publish
using FluxTraining

p = Publish.Project(FluxTraining)
rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)
deploy(FluxTraining; root = "/FluxTraining.jl", force = true, label = "dev")
